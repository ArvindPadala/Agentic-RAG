-- Apply to the dedicated AgenticRAG project. No service-role key is needed by
-- the application: every memory request runs as the verified user's JWT.
begin;

create table public.user_memories (
    user_id uuid primary key references auth.users(id) on delete cascade,
    memory jsonb not null,
    revision bigint not null check (revision > 0),
    is_closed boolean not null default false,
    updated_at timestamptz not null default now(),
    constraint closed_memory_is_scrubbed check (
        not is_closed or memory = '{"session_summaries":[],"preferences":{},"facts":[]}'::jsonb
    ),
    constraint user_memory_shape check (
        jsonb_typeof(memory) = 'object'
        and memory ?& array['session_summaries', 'preferences', 'facts']
        and memory - array['session_summaries', 'preferences', 'facts'] = '{}'::jsonb
        and jsonb_typeof(memory->'session_summaries') = 'array'
        and jsonb_typeof(memory->'preferences') = 'object'
        and jsonb_typeof(memory->'facts') = 'array'
        and jsonb_array_length(memory->'session_summaries') <= 10
        and jsonb_array_length(memory->'facts') <= 20
        and octet_length(memory::text) <= 16384
    )
);

alter table public.user_memories enable row level security;
revoke all on public.user_memories from public, anon, authenticated;
grant select, insert, update on public.user_memories to authenticated;

-- Supabase anonymous AUTH users are authenticated identities with their own
-- auth.uid(). They are not the unverified "anon" API role.
create policy user_memory_read on public.user_memories for select to authenticated
    using ((select auth.uid()) = user_id);
create policy user_memory_insert on public.user_memories for insert to authenticated
    with check ((select auth.uid()) = user_id);
create policy user_memory_update on public.user_memories for update to authenticated
    using ((select auth.uid()) = user_id and not is_closed)
    with check ((select auth.uid()) = user_id);
-- No DELETE grant/policy: removing a tombstone would allow a replayed JWT or
-- an in-flight revision-0 save to recreate personal memory after Forget.

create function public.save_user_memory(expected_revision bigint, new_memory jsonb)
returns setof public.user_memories
language plpgsql
security invoker
set search_path = ''
as $$
declare
    owner uuid := auth.uid();
    saved public.user_memories;
begin
    if owner is null then
        raise exception 'Authentication required' using errcode = '42501';
    end if;
    if expected_revision is null or expected_revision < 0 then
        raise exception 'Invalid revision' using errcode = '22023';
    end if;
    if expected_revision = 0 then
        insert into public.user_memories(user_id, memory, revision)
        values (owner, new_memory, 1)
        on conflict (user_id) do nothing
        returning * into saved;
    else
        update public.user_memories
        set memory = new_memory, revision = revision + 1, updated_at = now()
        where user_id = owner and revision = expected_revision and not is_closed
        returning * into saved;
    end if;
    if saved.user_id is null then
        -- PostgREST maps serialization_failure to HTTP 409. Never overwrite a
        -- concurrent session's memory or automatically repeat the LLM call.
        raise exception 'Memory revision conflict' using errcode = '40001';
    end if;
    return next saved;
end;
$$;

revoke all on function public.save_user_memory(bigint, jsonb) from public, anon;
grant execute on function public.save_user_memory(bigint, jsonb) to authenticated;

create function public.forget_user_memory()
returns uuid
language plpgsql
security invoker
set search_path = ''
as $$
declare
    owner uuid := auth.uid();
begin
    if owner is null then
        raise exception 'Authentication required' using errcode = '42501';
    end if;
    insert into public.user_memories(user_id, memory, revision, is_closed)
    values (owner, '{"session_summaries":[],"preferences":{},"facts":[]}', 1, true)
    on conflict (user_id) do update
        set memory = excluded.memory, is_closed = true,
            revision = public.user_memories.revision + 1, updated_at = now()
        where not public.user_memories.is_closed;
    return owner;
end;
$$;
revoke all on function public.forget_user_memory() from public, anon;
grant execute on function public.forget_user_memory() to authenticated;

commit;
