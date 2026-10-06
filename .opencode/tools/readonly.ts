import { tool, type ToolContext } from "@opencode-ai/plugin"
import { spawn } from "node:child_process"
import { mkdtemp, realpath, rm } from "node:fs/promises"
import { homedir, tmpdir } from "node:os"
import { dirname, join, relative, isAbsolute, sep } from "node:path"

// Explicit installation trees, never /System (which includes the Data volume),
// /Library, /opt, /private/var, /usr/local, or a user's entire home directory.
const runtimeTrees = [
  "/bin", "/sbin", "/usr/bin", "/usr/sbin", "/usr/lib", "/usr/libexec", "/usr/share",
  "/System/Library", "/System/Cryptexes/OS/usr/lib", "/System/Cryptexes/OS/System/Library",
  "/System/Volumes/Preboot/Cryptexes/OS/usr/lib",
  "/System/Volumes/Preboot/Cryptexes/OS/System/Library",
  "/Library/Apple", "/Library/Developer/CommandLineTools",
  "/Library/Frameworks/Python.framework", "/Applications/Xcode.app/Contents/Developer",
  "/private/var/db/dyld",
  "/opt/homebrew/bin", "/opt/homebrew/lib", "/opt/homebrew/libexec",
  "/opt/homebrew/Cellar", "/opt/homebrew/opt", "/opt/homebrew/share",
  "/usr/local/bin", "/usr/local/lib", "/usr/local/libexec", "/usr/local/Cellar",
  "/usr/local/opt", "/usr/local/share",
  "/opt/anaconda3/bin", "/opt/anaconda3/lib", "/opt/anaconda3/libexec", "/opt/anaconda3/share",
]
const runtimeFiles = [
  // dyld opens the filesystem root during process startup. This is a literal
  // directory grant, not a subpath grant to the rest of the filesystem.
  "/",
  "/dev/null", "/dev/zero", "/dev/random", "/dev/urandom",
  "/private/etc/localtime", "/private/etc/hosts", "/private/etc/protocols",
  "/private/etc/services", "/private/etc/ssl/cert.pem",
  "/private/var/select/sh", "/private/var/select/developer_dir",
]

function contains(parent: string, child: string) {
  const path = relative(parent, child)
  return path === "" || (path !== ".." && !path.startsWith(`..${sep}`) && !isAbsolute(path))
}

async function readRules(root: string, scratch: string) {
  const home = await realpath(homedir())
  if (contains(root, home)) throw new Error("repository root must not expose the entire home directory")
  const trees = [root, scratch]
  const files: string[] = []
  for (const [paths, resolved] of [[runtimeTrees, trees], [runtimeFiles, files]]) {
    for (const path of paths) {
      // Also allow querying a missing literal so runtime selectors can fall
      // back normally; this never grants access to its parent directory.
      if (resolved === files) files.push(path)
      try {
        const canonical = await realpath(path)
        // Do not turn a runtime-installation symlink into a home-directory grant.
        if (contains(home, canonical) || (resolved === trees && contains(canonical, home))) {
          throw new Error("runtime allowlist resolves into a user home directory")
        }
        const knownRuntime = runtimeTrees.some((tree) => contains(tree, canonical))
        const timezoneFile = path === "/private/etc/localtime" && contains("/private/var/db/timezone", canonical)
        if (canonical !== path && !knownRuntime && !timezoneFile) {
          throw new Error(`runtime allowlist resolves outside approved installation paths: ${path} -> ${canonical}`)
        }
        resolved.push(canonical)
      } catch (error) {
        if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error
      }
    }
  }
  // Only ancestor metadata is needed for pathname traversal/getcwd. Do not
  // grant directory listings or file contents outside the allowed trees.
  const ancestors = new Set<string>(["/var", "/etc", "/tmp"])
  for (const path of [...trees, ...files, ...runtimeTrees, ...runtimeFiles]) {
    for (let parent = dirname(path); ; parent = dirname(parent)) {
      ancestors.add(parent)
      if (parent === dirname(parent)) break
    }
  }
  return `(deny file-read*)
    (allow file-read* ${trees.map((path) => `(subpath ${JSON.stringify(path)})`).join(" ")}
      ${files.map((path) => `(literal ${JSON.stringify(path)})`).join(" ")})
    (allow file-read-metadata ${[...ancestors].map((path) => `(literal ${JSON.stringify(path)})`).join(" ")})`
}

const gitCommands = [
  "git status", "git status --short", "git status --porcelain",
  "git diff --no-ext-diff --no-textconv",
  "git diff --no-ext-diff --no-textconv --cached",
  "git log --oneline", "git log --oneline -5", "git log --oneline -10",
] as const

async function run(command: string, context: ToolContext, timeout: number, artifacts: boolean) {
  // Fail closed: never substitute an unsandboxed process on unsupported systems.
  if (process.platform !== "darwin") return "BLOCKED: read-only sandbox requires macOS sandbox-exec."
  const root = await realpath(context.worktree)
  const cwd = await realpath(context.directory)
  if (!contains(root, cwd)) return "BLOCKED: working directory must be inside the repository."
  const scratch = await realpath(await mkdtemp(join(tmpdir(), "opencode-readonly-")))
  if (contains(root, scratch)) {
    await rm(scratch, { recursive: true, force: true })
    return "BLOCKED: sandbox artifact directory must be outside the worktree."
  }
  try {
    const reads = await readRules(root, scratch)
    const profile = `(version 1)
    (allow default)
    ${reads}
    (deny file-write*)
    (deny network*)
    (deny mach-lookup)
    (deny appleevent-send)
    (deny signal)
    (allow signal (target same-sandbox))
    (allow file-write* (literal "/dev/null"))
    ${artifacts ? `(allow file-write* (subpath ${JSON.stringify(scratch)}))` : ""}`
    return await new Promise<string>((resolve) => {
      const child = spawn("/usr/bin/sandbox-exec", ["-p", profile, "/bin/sh", "-c", command], {
        cwd,
        detached: true,
        stdio: ["ignore", "pipe", "pipe"],
        env: {
          // Do not inherit credentials, user startup files, loader overrides,
          // external PYTHONPATH/NODE_PATH, or arbitrary user PATH entries.
          PATH: [join(root, ".venv/bin"), join(root, "venv/bin"), "/opt/anaconda3/bin",
            "/opt/homebrew/bin", "/usr/local/bin", "/usr/bin", "/bin", "/usr/sbin", "/sbin"].join(":"),
          HOME: scratch, CFFIXED_USER_HOME: scratch,
          LANG: "en_US.UTF-8", LC_ALL: "en_US.UTF-8",
          TMPDIR: scratch, TMP: scratch, TEMP: scratch,
          XDG_CACHE_HOME: scratch, XDG_CONFIG_HOME: scratch, XDG_DATA_HOME: scratch,
          PYTHONPYCACHEPREFIX: join(scratch, "pycache"), PYTHONNOUSERSITE: "1",
          PYTHONDONTWRITEBYTECODE: "1", GIT_OPTIONAL_LOCKS: "0", GIT_PAGER: "cat",
          PAGER: "cat", npm_config_cache: join(scratch, "npm"),
        },
      })
      let output = ""
      let truncated = false
      let stopped = ""
      const append = (chunk: Buffer) => {
        const text = chunk.toString()
        const remaining = 50_000 - output.length
        output += text.slice(0, Math.max(0, remaining))
        if (text.length > remaining) truncated = true
      }
      const kill = () => {
        if (child.pid) {
          try { process.kill(-child.pid, "SIGKILL") } catch { /* already exited */ }
        }
      }
      const abort = () => { stopped = "ABORTED"; kill() }
      const timer = setTimeout(() => { stopped = "TIMEOUT"; kill() }, timeout)
      context.abort.addEventListener("abort", abort, { once: true })
      child.stdout.on("data", append)
      child.stderr.on("data", append)
      child.on("error", (error) => { stopped = `BLOCKED: ${error.message}` })
      child.on("close", (code, signal) => {
        clearTimeout(timer)
        context.abort.removeEventListener("abort", abort)
        kill()
        resolve(`Command: ${command}\nExit: ${code ?? signal ?? "unavailable"}${stopped ? ` (${stopped})` : ""}\n${output}${truncated ? "\n[output truncated]" : ""}`)
      })
      if (context.abort.aborted) abort()
    })
  } catch (error) {
    return `BLOCKED: unable to enforce the read-only sandbox: ${(error as Error).message}`
  } finally {
    await rm(scratch, { recursive: true, force: true })
  }
}

export const git = tool({
  description: "Inspect status, staged/unstaged diffs, or recent history using exact read-only git commands in a filesystem-write-denying sandbox.",
  args: { command: tool.schema.enum(gitCommands) },
  async execute({ command }, context) {
    if (!["shipcheck", "deep-review", "production-reviewer"].includes(context.agent)) {
      return "BLOCKED: tool is limited to the read-only workflow agents."
    }
    if (!gitCommands.includes(command)) return "BLOCKED: unsupported git command."
    return run(command, context, 30_000, false)
  },
})

export const verify = tool({
  description: "Run verification in a macOS OS-enforced sandbox. Reads are restricted to the repository, disposable temp directory, and explicit system/runtime paths. Only temp is writable; network is denied. Fails closed.",
  args: {
    command: tool.schema.string().min(1),
    timeout: tool.schema.number().int().min(1_000).max(600_000).default(120_000),
  },
  async execute({ command, timeout }, context) {
    if (context.agent !== "shipcheck") return "BLOCKED: verification is limited to shipcheck."
    return run(command, context, timeout, true)
  },
})
