-- Run with Supabase CLI/pgTAP against a disposable local Supabase database.
-- All fixtures are synthetic, and the transaction rolls back.
begin;
select plan(28);

insert into auth.users (id, email) values
    ('11111111-1111-1111-1111-111111111111', 'a@example.invalid'),
    ('22222222-2222-2222-2222-222222222222', 'b@example.invalid'),
    ('33333333-3333-3333-3333-333333333333', 'c@example.invalid');

set local role anon;
select throws_ok($$select * from public.user_memories$$, '42501', null, 'anon cannot read memory');
select throws_ok($$insert into public.user_memories(user_id,memory,revision) values
    ('11111111-1111-1111-1111-111111111111', '{}', 1)$$,
    '42501', null, 'anon cannot write memory');
select throws_ok($$select * from public.save_user_memory(0, '{}')$$,
    '42501', null, 'anon cannot call save RPC');
select throws_ok($$select public.forget_user_memory()$$,
    '42501', null, 'anon cannot call forget RPC');

set local role authenticated;
set local request.jwt.claim.sub = '11111111-1111-1111-1111-111111111111';
select results_eq($$select revision from public.save_user_memory(0,
    '{"session_summaries":[],"preferences":{},"facts":["A"]}')$$,
    array[1::bigint], 'A creates own memory');
select results_eq($$select memory->'facts' from public.user_memories$$,
    array['["A"]'::jsonb], 'A reads own memory');

set local request.jwt.claim.sub = '22222222-2222-2222-2222-222222222222';
select is_empty($$select * from public.user_memories$$, 'B cannot read A');
select is_empty($$update public.user_memories set memory = '{}' returning user_id$$,
    'B cannot update A');
select throws_ok($$insert into public.user_memories(user_id,memory,revision) values
    ('11111111-1111-1111-1111-111111111111',
    '{"session_summaries":[],"preferences":{},"facts":[]}', 1)$$,
    '42501', null, 'B cannot create memory for A');
select results_eq($$select revision from public.save_user_memory(0,
    '{"session_summaries":[],"preferences":{},"facts":["B"]}')$$,
    array[1::bigint], 'B creates only B memory');
select throws_ok($$update public.user_memories
    set user_id='33333333-3333-3333-3333-333333333333'$$,
    '42501', null, 'B cannot reassign ownership');

set local request.jwt.claim.sub = '11111111-1111-1111-1111-111111111111';
select results_eq($$select memory->'facts' from public.user_memories$$,
    array['["A"]'::jsonb], 'denied update left A intact');
select throws_ok($$select * from public.save_user_memory(0,
    '{"session_summaries":[],"preferences":{},"facts":["stale"]}')$$,
    '40001', null, 'stale revision cannot overwrite A');
select results_eq($$select revision from public.save_user_memory(1,
    '{"session_summaries":[],"preferences":{},"facts":["updated A"]}')$$,
    array[2::bigint], 'valid update increments revision');
select results_eq($$select memory->'facts' from public.user_memories$$,
    array['["updated A"]'::jsonb], 'A update is persisted');

set local request.jwt.claim.sub = '22222222-2222-2222-2222-222222222222';
select throws_ok($$delete from public.user_memories
    where user_id='11111111-1111-1111-1111-111111111111'$$,
    '42501', null, 'B cannot delete A or remove tombstones');
set local request.jwt.claim.sub = '11111111-1111-1111-1111-111111111111';
select results_eq($$select revision from public.user_memories$$,
    array[2::bigint], 'denied delete left A intact');
select results_eq($$select public.forget_user_memory()::text$$,
    array['11111111-1111-1111-1111-111111111111'], 'A can forget own memory');
select results_eq($$select memory->'facts' from public.user_memories$$,
    array['[]'::jsonb], 'forgotten personal memory is scrubbed');
select results_eq($$select is_closed from public.user_memories$$,
    array[true], 'forgotten identity has a closed marker');
select results_eq($$select public.forget_user_memory()::text$$,
    array['11111111-1111-1111-1111-111111111111'], 'repeated Forget is idempotent');
select throws_ok($$select * from public.save_user_memory(2,
    '{"session_summaries":[],"preferences":{},"facts":["in flight"]}')$$,
    '40001', null, 'in-flight save cannot restore forgotten memory');
select throws_ok($$select * from public.save_user_memory(0,
    '{"session_summaries":[],"preferences":{},"facts":["replayed"]}')$$,
    '40001', null, 'revision-0 replay cannot resurrect a closed identity');
select is_empty($$update public.user_memories set is_closed=false returning user_id$$,
    'closed identity cannot reopen through direct API update');
select throws_ok($$delete from public.user_memories$$,
    '42501', null, 'owner cannot remove tombstone through direct API');
set local request.jwt.claim.sub = '22222222-2222-2222-2222-222222222222';
select results_eq($$select memory->'facts' from public.user_memories$$,
    array['["B"]'::jsonb], 'A deletion preserves B memory');

set local request.jwt.claim.sub = '33333333-3333-3333-3333-333333333333';
select results_eq($$select public.forget_user_memory()::text$$,
    array['33333333-3333-3333-3333-333333333333'], 'forgetting a new unsaved identity creates a tombstone');
select throws_ok($$select * from public.save_user_memory(0,
    '{"session_summaries":[],"preferences":{},"facts":["initial save after forget"]}')$$,
    '40001', null, 'initial in-flight save cannot recreate an unsaved forgotten identity');

select * from finish();
rollback;
