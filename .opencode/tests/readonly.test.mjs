import assert from "node:assert/strict"
import { execFileSync } from "node:child_process"
import { mkdir, mkdtemp, readFile, realpath, rm, writeFile } from "node:fs/promises"
import { randomUUID } from "node:crypto"
import { homedir, tmpdir } from "node:os"
import { dirname, join } from "node:path"
import { test } from "node:test"
import { git, verify } from "../tools/readonly.ts"

const context = (agent) => ({
  agent, directory: process.cwd(), worktree: process.cwd(),
  abort: new AbortController().signal,
})
const resolved = (name) => JSON.parse(execFileSync("opencode", ["debug", "agent", name], { encoding: "utf8" }))
const match = (pattern, value) => new RegExp(`^${pattern.split("*").map((part) => part.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join(".*")}$`, "s").test(value)
const permission = (agent, tool, target = "*") => agent.permission.filter((rule) =>
  match(rule.permission, tool) && match(rule.pattern, target)).at(-1)?.action
const python = (script) => `python3 -c '${script.replaceAll("'", "'\\''")}'`

test("resolved agents deny mutation, unrestricted execution, and delegation escape routes", () => {
  for (const name of ["shipcheck", "deep-review", "production-reviewer"]) {
    const agent = resolved(name)
    for (const tool of ["edit", "write", "apply_patch", "bash", "skill", "mcp_writer", "unknown_tool", "plan_enter", "plan_exit"]) {
      assert.equal(permission(agent, tool), "deny", `${name}: ${tool}`)
    }
    for (const name of ["build", "plan", "general", "explore", "deep-review"]) {
      assert.equal(permission(agent, "task", name), "deny")
    }
    assert.equal(permission(agent, "readonly_git"), "allow")
    assert.equal(permission(agent, "readonly_verify"), agent.name === "shipcheck" ? "allow" : "deny")
    assert.equal(permission(agent, "task", "production-reviewer"), agent.name === "shipcheck" ? "allow" : "deny")
    assert.equal(agent.model.providerID, "openai")
    assert.equal(agent.model.modelID, name === "deep-review" ? "gpt-6-astra" : "gpt-6.1-sol")
  }
})

test("commands preserve models, bind restricted agents, and retain shipping review", async () => {
  const config = JSON.parse(execFileSync("opencode", ["debug", "config"], { encoding: "utf8", maxBuffer: 10_000_000 }))
  for (const [name, model] of [["shipcheck", "gpt-6.1-sol"], ["deep-review", "gpt-6-astra"]]) {
    assert.equal(config.command[name].agent, name)
    assert.equal(config.command[name].model, `openai/${model}`)
    const text = await readFile(`.opencode/commands/${name}.md`, "utf8")
    assert.match(text, new RegExp(`^agent: ${name}$`, "m"))
    assert.match(text, new RegExp(`^model: openai/${model.replaceAll(".", "\\.")}$`, "m"))
    if (name === "shipcheck") {
      assert.match(text, /production-reviewer/)
      assert.match(text, /READY TO SHIP/)
      assert.match(text, /NOT READY TO SHIP/)
    }
  }
})

test("tool implementation rejects unauthorized agents and non-allowlisted git commands", async () => {
  assert.match(await verify.execute({ command: "true", timeout: 1000 }, context("deep-review")), /BLOCKED/)
  assert.match(await git.execute({ command: "git status; touch injected" }, context("shipcheck")), /BLOCKED/)
  assert.match(await git.execute({ command: "git status" }, context("build")), /BLOCKED/)
})

test("unsupported platforms fail closed rather than running unrestricted commands", async () => {
  const descriptor = Object.getOwnPropertyDescriptor(process, "platform")
  try {
    Object.defineProperty(process, "platform", { value: "linux", configurable: true })
    assert.match(await verify.execute({ command: "true", timeout: 1000 }, context("shipcheck")), /BLOCKED/)
    assert.match(await git.execute({ command: "git status" }, context("deep-review")), /BLOCKED/)
  } finally {
    Object.defineProperty(process, "platform", descriptor)
  }
})

test("sandbox supports inspection, verification, and disposable artifact writes", async () => {
  assert.match(await git.execute({ command: "git diff --no-ext-diff --no-textconv --cached" }, context("production-reviewer")), /Exit: 0/)
  const result = await verify.execute({ command: 'python3 -c \'import os, pathlib; p=pathlib.Path(os.environ["TMPDIR"])/"artifact"; p.write_text("ok"); print(p.read_text())\'', timeout: 10000 }, context("shipcheck"))
  assert.match(result, /Exit: 0/)
  assert.match(result, /\nok\s*$/)
})

test("normal repository reads, runtime imports, and make verification still work", async () => {
  const script = `import pathlib, unittest
class RepositoryRead(unittest.TestCase):
    def test_makefile(self):
        self.assertIn("python -m pytest tests/", pathlib.Path("Makefile").read_text())
        self.assertTrue(pathlib.Path(".opencode/commands/shipcheck.md").read_text())
unittest.main()`
  const result = await verify.execute({ command: python(script), timeout: 10000 }, context("shipcheck"))
  assert.match(result, /Exit: 0/)
  assert.match(result, /Ran 1 test/)
  const make = await verify.execute({ command: "make -n test", timeout: 10000 }, context("shipcheck"))
  assert.match(make, /Exit: 0/)
  assert.match(make, /python -m pytest tests\//)
})

test("external sentinel secrets cannot be read directly, through symlinks, hardlinks, or child processes", async () => {
  const fixture = await realpath(await mkdtemp(join(tmpdir(), "opencode-secret-test-")))
  const secret = `sentinel-${randomUUID()}`
  const paths = [".ssh/id_ed25519", ".aws/credentials", ".config/service/credentials",
    "Library/Application Support/Browser/Local State", "Projects/unrelated/private.txt", "arbitrary-secret"]
  const files = paths.map((path) => join(fixture, path))
  try {
    for (const file of files) {
      await mkdir(dirname(file), { recursive: true })
      await writeFile(file, secret)
      // Prove the Data-volume aliases exist before testing their denial.
      assert.equal(await readFile(`/System/Volumes/Data${file}`, "utf8"), secret)
    }
    const script = `import os, pathlib, subprocess
paths = ${JSON.stringify(files)}
scratch = pathlib.Path(os.environ["TMPDIR"])
for index, name in enumerate(paths):
    target = pathlib.Path(name)
    try:
        target.read_bytes()
        raise AssertionError("direct read escaped sandbox")
    except PermissionError:
        pass
    link = scratch / ("symlink-" + str(index))
    link.symlink_to(target)
    try:
        link.read_bytes()
        raise AssertionError("symlink read escaped sandbox")
    except PermissionError:
        pass
    hardlink = scratch / ("hardlink-" + str(index))
    try:
        os.link(target, hardlink)
        hardlink.read_bytes()
        raise AssertionError("hardlink read escaped sandbox")
    except PermissionError:
        pass
    child = subprocess.run(["/bin/cat", str(target)], capture_output=True)
    assert child.returncode != 0 and not child.stdout
    alias = pathlib.Path("/System/Volumes/Data" + name)
    try:
        alias.read_bytes()
        raise AssertionError("Data-volume alias escaped sandbox")
    except PermissionError:
        pass
print("all external secret reads denied")`
    const result = await verify.execute({ command: python(script), timeout: 10000 }, context("shipcheck"))
    assert.match(result, /Exit: 0/)
    assert.match(result, /all external secret reads denied/)
    assert.ok(!result.includes(secret))
    for (const file of files) assert.equal(await readFile(file, "utf8"), secret)
  } finally {
    await rm(fixture, { recursive: true, force: true })
  }
})

test("verification does not inherit user credentials or startup/loader overrides", async () => {
  const keys = ["READONLY_TEST_SECRET", "BASH_ENV", "ENV", "PYTHONPATH", "NODE_OPTIONS"]
  const original = keys.map((key) => process.env[key])
  try {
    for (const key of keys) process.env[key] = "not-for-verification"
    const script = `import os
assert not any(key in os.environ for key in ${JSON.stringify(keys)})
assert os.environ["HOME"] == os.environ["TMPDIR"]
assert os.environ["XDG_CONFIG_HOME"] == os.environ["TMPDIR"]
print("clean environment")`
    const result = await verify.execute({ command: python(script), timeout: 10000 }, context("shipcheck"))
    assert.match(result, /Exit: 0/)
    assert.match(result, /clean environment/)
  } finally {
    keys.forEach((key, index) => {
      if (original[index] === undefined) delete process.env[key]
      else process.env[key] = original[index]
    })
  }
})

test("out-of-repository working directories and roots exposing the whole home fail closed", async () => {
  assert.match(await verify.execute({ command: "true", timeout: 1000 }, {
    ...context("shipcheck"), directory: tmpdir(),
  }), /BLOCKED/)
  assert.match(await verify.execute({ command: "true", timeout: 1000 }, {
    ...context("shipcheck"), worktree: homedir(),
  }), /BLOCKED/)
})

test("sandbox blocks source write access and direct, subprocess, symlink, and deletion attacks", async () => {
  const fixture = await mkdtemp(join(tmpdir(), "opencode-permission-test-"))
  const sentinel = join(fixture, "sentinel")
  await writeFile(sentinel, "unchanged")
  try {
    const script = `import os, pathlib, subprocess
p = pathlib.Path(${JSON.stringify(sentinel)})
source = pathlib.Path(${JSON.stringify(join(process.cwd(), "agent.py"))})
attempts = [lambda: source.open("r+"), lambda: p.write_text("overwritten"), lambda: p.unlink(), lambda: p.rename(p.with_name("renamed"))]
for attempt in attempts:
    try:
        attempt()
        raise AssertionError("write permission escaped sandbox")
    except PermissionError:
        pass
link = pathlib.Path(os.environ["TMPDIR"])/"link"
link.symlink_to(p)
try:
    link.write_text("symlink overwrite")
    raise AssertionError("symlink escaped sandbox")
except PermissionError:
    pass
child = subprocess.run(["/bin/sh", "-c", "printf overwritten > \\\"$1\\\"", "sh", str(p)], capture_output=True)
assert child.returncode != 0
print("all writes denied")`
    const quoted = `'${script.replaceAll("'", "'\\''")}'`
    const result = await verify.execute({ command: `python3 -c ${quoted}`, timeout: 10000 }, context("shipcheck"))
    assert.match(result, /Exit: 0/)
    assert.match(result, /all writes denied/)
    assert.equal(await readFile(sentinel, "utf8"), "unchanged")
  } finally {
    await rm(fixture, { recursive: true, force: true })
  }
})

test("sandbox denies network and terminates timed-out checks", async () => {
  const result = await verify.execute({ command: "python3 -c 'import socket; socket.socket().connect((\"127.0.0.1\", 9))'", timeout: 10000 }, context("shipcheck"))
  assert.match(result, /Operation not permitted/)
  assert.match(await verify.execute({ command: "sleep 10", timeout: 1000 }, context("shipcheck")), /TIMEOUT/)
})
