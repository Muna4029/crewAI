from importlib.metadata import version as get_version
import importlib

import click

# Lazily import internal submodules to avoid triggering eager imports that
# may fail at import time (for example, modules under crewai.tools).
# These LazyLoader objects will import the real object on first use and
# replace themselves in globals(), preventing import-time failures.


class _LazyLoader:
    def __init__(self, module_path: str, attr_name: str, global_name: str):
        self._module_path = module_path
        self._attr_name = attr_name
        self._global_name = global_name
        self._loaded = False
        self._obj = None

    def _load(self):
        if not self._loaded:
            module = importlib.import_module(self._module_path)
            obj = getattr(module, self._attr_name)
            globals()[self._global_name] = obj
            self._obj = obj
            self._loaded = True
        return self._obj

    def __call__(self, *args, **kwargs):
        obj = self._load()
        return obj(*args, **kwargs)

    def __getattr__(self, item):
        obj = self._load()
        return getattr(obj, item)

    def __repr__(self):
        if self._loaded:
            return repr(self._obj)
        return f"<LazyLoader {self._module_path}.{self._attr_name}>"


# Lazy proxies for CLI subcommands and classes. These match the original
# import targets but defer importing the underlying modules until needed.
add_crew_to_flow = _LazyLoader(
    "crewai.cli.add_crew_to_flow", "add_crew_to_flow", "add_crew_to_flow"
)
create_crew = _LazyLoader("crewai.cli.create_crew", "create_crew", "create_crew")
create_flow = _LazyLoader("crewai.cli.create_flow", "create_flow", "create_flow")
run_chat = _LazyLoader("crewai.cli.crew_chat", "run_chat", "run_chat")
KickoffTaskOutputsSQLiteStorage = _LazyLoader(
    "crewai.memory.storage.kickoff_task_outputs_storage",
    "KickoffTaskOutputsSQLiteStorage",
    "KickoffTaskOutputsSQLiteStorage",
)

AuthenticationCommand = _LazyLoader(
    "crewai.cli.authentication.main", "AuthenticationCommand", "AuthenticationCommand"
)
DeployCommand = _LazyLoader("crewai.cli.deploy.main", "DeployCommand", "DeployCommand")
evaluate_crew = _LazyLoader(
    "crewai.cli.evaluate_crew", "evaluate_crew", "evaluate_crew"
)
install_crew = _LazyLoader("crewai.cli.install_crew", "install_crew", "install_crew")
kickoff_flow = _LazyLoader("crewai.cli.kickoff_flow", "kickoff_flow", "kickoff_flow")
plot_flow = _LazyLoader("crewai.cli.plot_flow", "plot_flow", "plot_flow")
replay_task_command = _LazyLoader(
    "crewai.cli.replay_from_task", "replay_task_command", "replay_task_command"
)
reset_memories_command = _LazyLoader(
    "crewai.cli.reset_memories_command",
    "reset_memories_command",
    "reset_memories_command",
)
run_crew = _LazyLoader("crewai.cli.run_crew", "run_crew", "run_crew")
ToolCommand = _LazyLoader("crewai.cli.tools.main", "ToolCommand", "ToolCommand")
train_crew = _LazyLoader("crewai.cli.train_crew", "train_crew", "train_crew")
update_crew = _LazyLoader("crewai.cli.update_crew", "update_crew", "update_crew")


@click.group()
@click.version_option(get_version("crewai"))
def crewai():
    """Top-level command group for crewai."""


@crewai.command()
@click.argument("type", type=click.Choice(["crew", "flow"]))
@click.argument("name")
@click.option("--provider", type=str, help="The provider to use for the crew")
@click.option("--skip_provider", is_flag=True, help="Skip provider validation")
def create(type, name, provider, skip_provider=False):
    """Create a new crew, or flow."""
    if type == "crew":
        create_crew(name, provider, skip_provider)
    elif type == "flow":
        create_flow(name)
    else:
        click.secho("Error: Invalid type. Must be 'crew' or 'flow'.", fg="red")


@crewai.command()
@click.option(
    "--tools", is_flag=True, help="Show the installed version of crewai tools"
)
def version(tools):
    """Show the installed version of crewai."""
    try:
        crewai_version = get_version("crewai")
    except Exception:
        crewai_version = "unknown version"
    click.echo(f"crewai version: {crewai_version}")

    if tools:
        try:
            tools_version = get_version("crewai")
            click.echo(f"crewai tools version: {tools_version}")
        except Exception:
            click.echo("crewai tools not installed")


@crewai.command()
@click.option(
    "-n",
    "--n_iterations",
    type=int,
    default=5,
    help="Number of iterations to train the crew",
)
@click.option(
    "-f",
    "--filename",
    type=str,
    default="trained_agents_data.pkl",
    help="Path to a custom file for training",
)
def train(n_iterations: int, filename: str):
    """Train the crew."""
    click.echo(f"Training the Crew for {n_iterations} iterations")
    train_crew(n_iterations, filename)


@crewai.command()
@click.option(
    "-t",
    "--task_id",
    type=str,
    help="Replay the crew from this task ID, including all subsequent tasks.",
)
def replay(task_id: str) -> None:
    """
    Replay the crew execution from a specific task.

    Args:
        task_id (str): The ID of the task to replay from.
    """
    try:
        click.echo(f"Replaying the crew from task {task_id}")
        replay_task_command(task_id)
    except Exception as e:
        click.echo(f"An error occurred while replaying: {e}", err=True)


@crewai.command()
def log_tasks_outputs() -> None:
    """
    Retrieve your latest crew.kickoff() task outputs.
    """
    try:
        storage = KickoffTaskOutputsSQLiteStorage()
        tasks = storage.load()

        if not tasks:
            click.echo(
                "No task outputs found. Only crew kickoff task outputs are logged."
            )
            return

        for index, task in enumerate(tasks, 1):
            click.echo(f"Task {index}: {task['task_id']}")
            click.echo(f"Description: {task['expected_output']}")
            click.echo("------")

    except Exception as e:
        click.echo(f"An error occurred while logging task outputs: {e}", err=True)


@crewai.command()
@click.option("-l", "--long", is_flag=True, help="Reset LONG TERM memory")
@click.option("-s", "--short", is_flag=True, help="Reset SHORT TERM memory")
@click.option("-e", "--entities", is_flag=True, help="Reset ENTITIES memory")
@click.option("-kn", "--knowledge", is_flag=True, help="Reset KNOWLEDGE storage")
@click.option(
    "-akn", "--agent-knowledge", is_flag=True, help="Reset AGENT KNOWLEDGE storage"
)
@click.option(
    "-k", "--kickoff-outputs", is_flag=True, help="Reset LATEST KICKOFF TASK OUTPUTS"
)
@click.option("-a", "--all", is_flag=True, help="Reset ALL memories")
def reset_memories(
    long: bool,
    short: bool,
    entities: bool,
    knowledge: bool,
    kickoff_outputs: bool,
    agent_knowledge: bool,
    all: bool,
) -> None:
    """
    Reset the crew memories (long, short, entity, latest_crew_kickoff_ouputs, knowledge, agent_knowledge). This will delete all the data saved.
    """
    try:
        memory_types = [
            long,
            short,
            entities,
            knowledge,
            agent_knowledge,
            kickoff_outputs,
            all,
        ]
        if not any(memory_types):
            click.echo(
                "Please specify at least one memory type to reset using the appropriate flags."
            )
            return
        reset_memories_command(
            long, short, entities, knowledge, agent_knowledge, kickoff_outputs, all
        )
    except Exception as e:
        click.echo(f"An error occurred while resetting memories: {e}", err=True)


@crewai.command()
@click.option(
    "-n",
    "--n_iterations",
    type=int,
    default=3,
    help="Number of iterations to Test the crew",
)
@click.option(
    "-m",
    "--model",
    type=str,
    default="gpt-4o-mini",
    help="LLM Model to run the tests on the Crew. For now only accepting only OpenAI models.",
)
def test(n_iterations: int, model: str):
    """Test the crew and evaluate the results."""
    click.echo(f"Testing the crew for {n_iterations} iterations with model {model}")
    evaluate_crew(n_iterations, model)


@crewai.command(
    context_settings=dict(
        ignore_unknown_options=True,
        allow_extra_args=True,
    )
)
@click.pass_context
def install(context):
    """Install the Crew."""
    install_crew(context.args)


@crewai.command()
def run():
    """Run the Crew."""
    run_crew()


@crewai.command()
def update():
    """Update the pyproject.toml of the Crew project to use uv."""
    update_crew()


@crewai.command()
def signup():
    """Sign Up/Login to CrewAI+."""
    AuthenticationCommand().signup()


@crewai.command()
def login():
    """Sign Up/Login to CrewAI+."""
    AuthenticationCommand().signup()


# DEPLOY CREWAI+ COMMANDS
@crewai.group()
def deploy():
    """Deploy the Crew CLI group."""
    pass


@crewai.group()
def tool():
    """Tool Repository related commands."""
    pass


@deploy.command(name="create")
@click.option("-y", "--yes", is_flag=True, help="Skip the confirmation prompt")
def deploy_create(yes: bool):
    """Create a Crew deployment."""
    deploy_cmd = DeployCommand()
    deploy_cmd.create_crew(yes)


@deploy.command(name="list")
def deploy_list():
    """List all deployments."""
    deploy_cmd = DeployCommand()
    deploy_cmd.list_crews()


@deploy.command(name="push")
@click.option("-u", "--uuid", type=str, help="Crew UUID parameter")
def deploy_push(uuid: Optional[str]):
    """Deploy the Crew."""
    deploy_cmd = DeployCommand()
    deploy_cmd.deploy(uuid=uuid)


@deploy.command(name="status")
@click.option("-u", "--uuid", type=str, help="Crew UUID parameter")
def deply_status(uuid: Optional[str]):
    """Get the status of a deployment."""
    deploy_cmd = DeployCommand()
    deploy_cmd.get_crew_status(uuid=uuid)


@deploy.command(name="logs")
@click.option("-u", "--uuid", type=str, help="Crew UUID parameter")
def deploy_logs(uuid: Optional[str]):
    """Get the logs of a deployment."""
    deploy_cmd = DeployCommand()
    deploy_cmd.get_crew_logs(uuid=uuid)


@deploy.command(name="remove")
@click.option("-u", "--uuid", type=str, help="Crew UUID parameter")
def deploy_remove(uuid: Optional[str]):
    """Remove a deployment."""
    deploy_cmd = DeployCommand()
    deploy_cmd.remove_crew(uuid=uuid)


@tool.command(name="create")
@click.argument("handle")
def tool_create(handle: str):
    tool_cmd = ToolCommand()
    tool_cmd.create(handle)


@tool.command(name="install")
@click.argument("handle")
def tool_install(handle: str):
    tool_cmd = ToolCommand()
    tool_cmd.login()
    tool_cmd.install(handle)


@tool.command(name="publish")
@click.option(
    "--force",
    is_flag=True,
    show_default=True,
    default=False,
    help="Bypasses Git remote validations",
)
@click.option("--public", "is_public", flag_value=True, default=False)
@click.option("--private", "is_public", flag_value=False)
def tool_publish(is_public: bool, force: bool):
    tool_cmd = ToolCommand()
    tool_cmd.login()
    tool_cmd.publish(is_public, force)


@crewai.group()
def flow():
    """Flow related commands."""
    pass


@flow.command(name="kickoff")
def flow_run():
    """Kickoff the Flow."""
    click.echo("Running the Flow")
    kickoff_flow()


@flow.command(name="plot")
def flow_plot():
    """Plot the Flow."""
    click.echo("Plotting the Flow")
    plot_flow()


@flow.command(name="add-crew")
@click.argument("crew_name")
def flow_add_crew(crew_name):
    """Add a crew to an existing flow."""
    click.echo(f"Adding crew {crew_name} to the flow")
    add_crew_to_flow(crew_name)


@crewai.command()
def chat():
    """
    Start a conversation with the Crew, collecting user-supplied inputs,
    and using the Chat LLM to generate responses.
    """
    click.secho(
        "\nStarting a conversation with the Crew\n" "Type 'exit' or Ctrl+C to quit.\n",
    )

    run_chat()


if __name__ == "__main__":
    crewai()
