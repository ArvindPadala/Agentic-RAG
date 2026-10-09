#!/usr/bin/env bash
# Disposable SQL verification; no published ports, host mounts, or hosted credentials.
set -euo pipefail

image="agenticrag-postgres-rls:test"
docker build -f tests/fixtures/Dockerfile.postgres -t "$image" tests/fixtures
container=$(docker run --rm -d --network none \
    -e POSTGRES_HOST_AUTH_METHOD=trust "$image")
trap 'docker stop "$container" >/dev/null' EXIT

for attempt in {1..30}; do
    if docker exec "$container" pg_isready -U postgres >/dev/null 2>&1; then
        break
    fi
    sleep 1
done
docker exec "$container" pg_isready -U postgres
docker exec "$container" psql -X -A -t -U postgres -c 'show server_version;'
docker exec -i "$container" psql -X -v ON_ERROR_STOP=1 -U postgres \
    < tests/fixtures/postgres_auth_fixture.sql
docker exec -i "$container" psql -X -v ON_ERROR_STOP=1 -U postgres \
    < supabase/migrations/202610070001_user_memory.sql
result=$(docker exec -i "$container" psql -X -A -t -v ON_ERROR_STOP=1 -U postgres \
    < supabase/tests/user_memory_rls.test.sql)

# pgTAP failures can leave psql's exit code zero. Validate the TAP output too.
python -c '
import re
import sys
text = sys.argv[1]
print(text)
lines = [line.strip() for line in text.splitlines()]
plans = [int(m.group(1)) for line in lines if (m := re.fullmatch(r"1\.\.(\d+)", line))]
passed = sum(line.startswith("ok ") for line in lines)
failed = any(line.startswith("not ok ") for line in lines)
if len(plans) != 1 or passed != plans[0] or failed:
    sys.exit("Database regression checks failed or did not complete")
print(f"Database regression checks: {passed} passed")
' "$result"
