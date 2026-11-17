# AI Agents for Nanoindentation (WG4)

A LangGraph multi-agent system for autonomous, objective-driven nanoindentation
experiments. A supervisor routes between hypothesis, literature, image-analysis,
measurement-strategy, observation and critic agents. Every instrument action
passes a human approval gate.

## Repository layout

| Path | Contents |
|---|---|
| `nanoindentation_agents/` | Agents, critics, LangGraph assembly, analysis, instrument session logic |
| `instrument_control/` | HTTP instrument client, instrument simulator and simulator server |
| `config/` | Default experiment, model, prompt and tool configuration (YAML) |
| `scripts/` | `run_agent.py` (CLI entry point), `run_sim_server.py` (simulator) |
| `notebooks/` | Example campaign notebooks (outputs cleared) |
| `literature/` | Short background notes used by the literature agent |
| `tests/` | pytest suite |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in HF_TOKEN; never commit .env
```

## Quick start (simulator, no hardware)

```bash
python scripts/run_sim_server.py        # terminal 1
python scripts/run_agent.py --check     # terminal 2: confirm connection
python scripts/run_agent.py -q "Is hardness higher near grain boundaries?"
```

Set `instrument.sim_mode: true` in `config/default.yaml`, or point
`--host/--port` at the simulator.

## Real instrument

The agents talk to an instrument-side HTTP server through
`instrument_control/instrument_client.py`. **That server is not included in this
repository.** Configure it through environment variables:

| Variable | Purpose |
|---|---|
| `NANOINDENT_HOST`, `NANOINDENT_PORT` | Instrument server address |
| `NANOINDENT_TOKEN` | Shared auth token (keep it in `.env` only) |
| `NANOINDENT_EXPORT_DIR` | InView export folder on the instrument PC |

## Tests

```bash
pytest
```

Tests that use real instrument frames or exports skip automatically when that
data is not present. The data is not distributed with the code.

## Data

Experimental data from real campaigns is not included in this repository. It
will be released separately with the associated publication.

## License

MIT. See [LICENSE](LICENSE). To cite this software, see [CITATION.cff](CITATION.cff).
