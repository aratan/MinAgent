# MinAgent resident

`minagent resident` runs the improvement loop with no terminal attached, so a
machine nobody is typing into can still improve. This directory holds the
systemd unit that starts it.

There is no orchestrator here on purpose. An earlier version of this directory
held ~490 lines of scheduling code that imported modules which never existed and
shipped a systemd example pointing at a file that was never written. It was
replaced by a unit file. The loop already knows how to wait until the machine is
free, budget its own work and stop cleanly; a second scheduler on top of that
would only be able to disagree with it.

## What it does

- Every `IMPROVEMENT_CYCLE_SECONDS`, checks whether the machine is idle: no
  logged-in user, no active session, CPU below the threshold, and no request of
  yours in flight.
- If idle, spends from a hard budget: reflect on recent sessions, propose one
  change, and have a reviewer check it before anything is written.
- A change that survives review goes into a trial, and is kept only if it beats
  the previous value on held-back cases. Otherwise it is reverted.

## What it does not do

- It does not watch you. No camera, no microphone, no continuous sensing.
  `src/minagent/senses.py` is the boundary.
- It does not spend without a cap. The budget is enforced before each call, and
  the spend survives the process.
- It does not run unless you turn it on: `IMPROVEMENT_AUTONOMOUS=on`.

## Install

```sh
# 1. Turn the loop on, in the project .env
echo 'IMPROVEMENT_AUTONOMOUS=on' >> .env

# 2. Prove it works by hand first. This runs in the foreground, no unit needed.
.venv/bin/python -m minagent resident
```

If that starts, prints `Improvement loop running...`, and stops on Ctrl-C, the
unit will work. Fix the two absolute paths in the unit if your checkout lives
somewhere else, then:

```sh
mkdir -p ~/.config/systemd/user
cp proactivo/minagent-resident.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now minagent-resident
```

## Operate

```sh
systemctl --user status minagent-resident
journalctl --user -u minagent-resident -f
systemctl --user stop minagent-resident
```

What it is doing, from inside MinAgent: the `/mejoras` command reports the loop's
state, and a trial in progress shows both arms' measured rates.

## After a login

The unit is a *user* unit, so it needs a session bus. Over SSH or in a container
there is no user manager, and `systemctl --user` will fail with a bus error. Two
options:

- `loginctl enable-linger "$USER"` once, so the user manager runs with no
  session at all. This is the normal fix.
- Or skip systemd and use cron, which has no such requirement. Cron is cruder -
  it cannot wait for a clean shutdown, so a run can be cut mid-cycle and the
  next start finds a half-written trial document - but the budget still caps
  what that costs.

```cron
*/15 * * * * cd /home/victor/proyecto/MinAgent && .venv/bin/python -m minagent resident
```

Use one or the other, not both: they would run two loops against the same
budget file, and they would not know about each other.
