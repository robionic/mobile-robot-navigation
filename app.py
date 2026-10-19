from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from navigation import (
    Algorithm,
    Simulation,
    apply_point_table,
    generate_sample_map,
    image_to_occupancy,
    find_path,
)


st.set_page_config(page_title="Mobile Robot Navigation Lab", page_icon="🤖", layout="wide")
st.markdown(
    """
    <style>
    .stApp { background: #f4f7fb; color: #172235; }
    [data-testid="stAppViewContainer"] { background: #f4f7fb; }
    [data-testid="stSidebar"] { background: #e8eef6; }
    [data-testid="stMetric"] { background: #ffffff; padding: 14px; border-radius: 12px; }
    div[data-testid="stVerticalBlock"] div:has(> div[data-testid="stMetric"]) {
        border-color: #d4deea;
    }
    [data-testid="stMarkdownContainer"], label, p, h1, h2, h3 { color: #172235; }
    div[data-testid="stAlert"] { color: #172235; }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("🤖 Mobile Robot Navigation Lab")
st.caption(
    "Explore grid-based planning, obstacle avoidance, and particle-swarm robot motion. "
    "Inspired by the linked Theta* / APF / Pure Pursuit navigation demonstration."
)

ALGORITHM_HELP = r"""
Choose the path-planning or steering method:

- **Theta\***: any-angle grid search. [Paper](https://doi.org/10.1613/jair.2994)
- **A\***: heuristic shortest-path search. [Paper](https://doi.org/10.1109/TSSC.1968.300136)
- **D\***: incremental replanning (implemented as D\* Lite). [Paper](https://idm-lab.org/bib/abstracts/papers/aaai02b.pdf)
- **RRT / Bi-RRT**: randomized single-query tree planners. [RRT paper](https://lavalle.pl/papers/Lav98c.pdf); [RRT-Connect paper](https://doi.org/10.1109/ROBOT.2000.844730)
- **PRM / PRM\***: sampled configuration-space roadmaps. [PRM paper](https://doi.org/10.1109/70.508439); [PRM\* paper](https://doi.org/10.1177/0278364911406761)
- **Ant colony / Firefly**: stochastic swarm-based path search/refinement. [Ant System paper](https://doi.org/10.1109/3477.484436); [Firefly paper](https://doi.org/10.1007/978-3-642-04944-6_14)
- **APF**: goal attraction and obstacle repulsion. [Paper](https://doi.org/10.1177/027836498600500106)
- **Pure Pursuit**: follow a route using a look-ahead target. [Technical report](https://www.ri.cmu.edu/pub_files/pub3/coulter_r_craig_1992_1/coulter_r_craig_1992_1.pdf)
"""

with st.sidebar:
    st.header("Map")
    map_source = st.selectbox(
        "Map source",
        ["Sample binary map", "Upload an image"],
        help="Use the built-in obstacle course or load a binary/grayscale image as an occupancy map.",
    )
    map_width = st.slider(
        "Map width (cells)", 30, 220, 100, 5,
        help="Horizontal grid resolution after creating or resizing the map. A cell is one robot-planning unit.",
    )
    map_height = st.slider(
        "Map height (cells)", 30, 180, 70, 5,
        help="Vertical grid resolution after creating or resizing the map. A cell is one robot-planning unit.",
    )
    uploaded_image = None
    threshold = 127
    dark_is_obstacle = True
    if map_source == "Upload an image":
        uploaded_image = st.file_uploader(
            "Binary / grayscale map image",
            type=["png", "jpg", "jpeg", "bmp", "tif"],
            help="Upload a map where one pixel color represents obstacles and the other represents free space.",
        )
        threshold = st.slider(
            "Obstacle threshold", 0, 255, 127,
            help="Grayscale cutoff for classifying pixels. Pixels on the obstacle side of this threshold become blocked.",
        )
        dark_is_obstacle = st.selectbox(
            "Obstacle pixels are",
            ["Dark pixels", "Bright pixels"],
            help="Select which side of the grayscale image represents occupied/blocked map cells.",
        ) == "Dark pixels"
    st.markdown("**Point table** — one `x,y,type` row per point; coordinates start at the top-left.")
    point_table = st.text_area(
        "Start, goal, and obstacle points",
        value="x,y,type",
        height=120,
        help="Optional CSV rows: `x,y,start`, `x,y,goal`, or `x,y,obstacle`. Coordinates are zero-based from the upper-left; obstacle rows block a cell.",
    )

    st.header("Navigation")
    algorithm: Algorithm = st.selectbox(
        "Algorithm",
        [
            "Theta*", "A*", "D*", "Bi-RRT", "RRT", "Ant colony",
            "Firefly", "PRM", "PRM*", "APF", "Pure Pursuit",
        ],
        help=ALGORITHM_HELP,
    )
    planner_options: dict[str, int | float] = {}
    if algorithm in {"RRT", "Bi-RRT"}:
        planner_options["planning_iterations"] = st.slider(
            "Planning iteration limit", 250, 10000, 2500, 250,
            help="Maximum random tree-growth attempts before reporting failure. Higher limits improve the chance of finding a route but take longer. [RRT](https://lavalle.pl/papers/Lav98c.pdf) · [RRT-Connect](https://doi.org/10.1109/ROBOT.2000.844730)",
        )
        planner_options["rrt_step_size"] = st.slider(
            "RRT extension step (cells)", 1.0, 12.0, 3.0, 0.5,
            help="Maximum distance in grid cells that a tree branch extends toward each random sample. Smaller steps fit narrow passages better but need more iterations. [RRT paper](https://lavalle.pl/papers/Lav98c.pdf)",
        )
        planner_options["goal_bias"] = st.slider(
            "RRT goal sampling probability", 0.0, 0.5, 0.12, 0.01,
            help="Chance that an RRT iteration samples the goal directly instead of a random free cell. Higher bias can speed convergence but reduce exploration. [RRT-Connect paper](https://doi.org/10.1109/ROBOT.2000.844730)",
        )
    elif algorithm in {"PRM", "PRM*"}:
        planner_options["roadmap_samples"] = st.slider(
            "Roadmap samples", 30, 1200, 180, 10,
            help="Number of randomly sampled free-space configurations added between start and goal. More samples improve connectivity at the cost of preprocessing time. [PRM](https://doi.org/10.1109/70.508439) · [PRM*](https://doi.org/10.1177/0278364911406761)",
        )
        if algorithm == "PRM":
            planner_options["roadmap_neighbors"] = st.slider(
                "Roadmap neighbors per sample", 2, 40, 12,
                help="For basic PRM, the maximum nearest nodes each sampled configuration tries to connect to when the straight segment is collision-free. [PRM paper](https://doi.org/10.1109/70.508439)",
            )
        else:
            planner_options["prm_star_gamma"] = st.slider(
                "PRM* neighbor coefficient", 1.0, 8.0, 2.0, 0.25,
                help="Scales the logarithmic nearest-neighbor count: ceil(coefficient × ln(samples + 2)). Larger values add more candidate edges and computation. [PRM* paper](https://doi.org/10.1177/0278364911406761)",
            )
    elif algorithm == "Ant colony":
        planner_options["aco_iterations"] = st.slider(
            "Ant-colony iterations", 5, 150, 35, 5,
            help="Number of pheromone-update rounds. More iterations give routes more time to improve. [Ant System paper](https://doi.org/10.1109/3477.484436)",
        )
        planner_options["aco_ants"] = st.slider(
            "Ants per iteration", 4, 120, 28, 2,
            help="Number of independent route-building agents per round. More ants explore more candidate paths but increase runtime. [Ant System paper](https://doi.org/10.1109/3477.484436)",
        )
        planner_options["aco_evaporation"] = st.slider(
            "Pheromone evaporation", 0.05, 0.8, 0.25, 0.05,
            help="Fraction of accumulated pheromone removed after each round. Higher evaporation reduces stale-route influence; lower evaporation preserves history. [Ant System paper](https://doi.org/10.1109/3477.484436)",
        )
        planner_options["aco_pheromone_exponent"] = st.slider(
            "Pheromone influence", 0.1, 4.0, 1.2, 0.1,
            help="Exponent α in the route-choice weight τ^α × η^β: controls how strongly ants prefer edges with accumulated pheromone τ. [Ant System paper](https://doi.org/10.1109/3477.484436)",
        )
        planner_options["aco_heuristic_exponent"] = st.slider(
            "Goal heuristic influence", 0.1, 6.0, 3.0, 0.1,
            help="Exponent β in τ^α × η^β, where η favors steps closer to the goal. Higher values make ants more goal-directed. [Ant System paper](https://doi.org/10.1109/3477.484436)",
        )
    elif algorithm == "Firefly":
        planner_options["firefly_population"] = st.slider(
            "Firefly population", 4, 60, 18, 2,
            help="Number of candidate route refinements that attract one another toward better routes. Larger populations explore more alternatives. [Firefly algorithm](https://doi.org/10.1007/978-3-642-04944-6_14)",
        )
        planner_options["firefly_iterations"] = st.slider(
            "Firefly iterations", 5, 150, 30, 5,
            help="Number of rounds in which candidate routes are attracted toward lower-cost routes. More rounds may improve route length but take longer. [Firefly algorithm](https://doi.org/10.1007/978-3-642-04944-6_14)",
        )
        planner_options["firefly_attraction"] = st.slider(
            "Firefly attraction", 0.05, 1.0, 0.35, 0.05,
            help="Fraction of the waypoint displacement toward a better candidate route per refinement attempt. Higher attraction moves routes more aggressively. [Firefly algorithm](https://doi.org/10.1007/978-3-642-04944-6_14)",
        )
        planner_options["firefly_randomness"] = st.slider(
            "Firefly randomness", 0.0, 3.0, 0.6, 0.1,
            help="Standard deviation of random waypoint perturbations in cells. Higher randomness explores more but can make refinement less stable. [Firefly algorithm](https://doi.org/10.1007/978-3-642-04944-6_14)",
        )
    particle_count = st.slider(
        "Number of particles", 1, 500, 80, 1,
        help="Number of independently noise-perturbed point-robot rollouts shown in the live simulation.",
    )
    max_steps = st.slider(
        "Maximum simulation steps", 50, 2000, 500, 50,
        help="Maximum number of motion updates before the run stops, even if particles have not reached the goal.",
    )
    render_every = st.slider(
        "Render every N steps", 1, 30, 5,
        help="Refresh the animated plot after this many motion updates. Larger values improve speed at the expense of animation smoothness.",
    )
    playback_delay = st.slider(
        "Playback delay (seconds)", 0.0, 0.20, 0.02, 0.01,
        help="Extra pause after each displayed frame. Set to zero for fastest playback.",
    )
    random_seed = st.number_input(
        "Random seed", min_value=0, max_value=1_000_000, value=7,
        help="Seed used by the stochastic planners and particle steering. Reuse a seed to reproduce a run.",
    )

    st.header("Robot & sensor")
    speed = st.slider(
        "Robot speed (cells/s)", 0.2, 8.0, 1.4, 0.1,
        help="Maximum distance a point robot can travel per second, measured in map cells.",
    )
    timestep = st.slider(
        "Time step (s)", 0.05, 1.0, 0.35, 0.05,
        help="Simulated time advanced per motion update. Displacement per step is speed multiplied by time step.",
    )
    noise = st.slider(
        "Particle steering noise", 0.0, 1.0, 0.18, 0.01,
        help="Standard deviation of random steering perturbation. Higher values spread the particle rollouts more.",
    )
    lookahead = st.slider(
        "Look-ahead distance (cells)", 1.0, 20.0, 5.0, 0.5,
        help="Distance ahead on the planned path used as the steering target by Pure Pursuit; in APF it adds route-guidance. [Pure Pursuit report](https://www.ri.cmu.edu/pub_files/pub3/coulter_r_craig_1992_1/coulter_r_craig_1992_1.pdf)",
    )
    lidar_range = st.slider(
        "LIDAR range (cells)", 1.0, 25.0, 7.0, 0.5,
        help="Maximum obstacle distance considered by the APF repulsive-force calculation. [Potential-field method](https://doi.org/10.1177/027836498600500106)",
    )
    attraction = st.slider(
        "APF goal attraction", 0.1, 5.0, 1.0, 0.1,
        help="Weight of the force pulling the robot toward the goal in the Artificial Potential Field controller. [Khatib, 1986](https://doi.org/10.1177/027836498600500106)",
    )
    repulsion = st.slider(
        "APF obstacle repulsion", 0.0, 10.0, 2.5, 0.1,
        help="Weight of the force pushing the robot away from obstacles detected within LIDAR range. [Khatib, 1986](https://doi.org/10.1177/027836498600500106)",
    )
    goal_tolerance = st.slider(
        "Goal tolerance (cells)", 0.5, 8.0, 2.0, 0.5,
        help="A particle is marked as having reached the goal when it comes within this distance.",
    )
    run = st.button("▶ Run live simulation", type="primary", width="stretch")

try:
    if map_source == "Sample binary map":
        occupancy = generate_sample_map(map_width, map_height)
    elif uploaded_image is not None:
        occupancy = image_to_occupancy(
            uploaded_image.getvalue(), map_width, map_height,
            threshold=threshold, dark_is_obstacle=dark_is_obstacle,
        )
    else:
        st.info("Upload an image, or switch to the included sample binary map.")
        st.stop()
    occupancy, start, goal = apply_point_table(occupancy, point_table)
except (ValueError, OSError) as error:
    st.error(str(error))
    st.stop()

preview_columns = st.columns([1.25, 1])
with preview_columns[0]:
    st.subheader("Occupancy map")
    fig_map, ax_map = plt.subplots(figsize=(8, 4.7))
    ax_map.imshow(occupancy, cmap="gray_r", origin="upper", interpolation="nearest")
    ax_map.scatter(*start, c="#27d7a1", marker="o", s=90, edgecolors="white", label="Start")
    ax_map.scatter(*goal, c="#ff5964", marker="*", s=150, edgecolors="white", label="Goal")
    ax_map.set(xlim=(-0.5, map_width - 0.5), ylim=(map_height - 0.5, -0.5))
    ax_map.set_aspect("equal")
    ax_map.legend(loc="upper right", facecolor="white", labelcolor="#172235")
    ax_map.set_facecolor("white")
    fig_map.tight_layout()
    st.pyplot(fig_map, width="stretch")
    plt.close(fig_map)

with preview_columns[1]:
    st.subheader("Planner preview")
    try:
        preview_path = find_path(
            occupancy, start, goal, algorithm,
            options=planner_options, seed=int(random_seed),
        )
        st.success(f"{algorithm} is ready · {len(preview_path)} waypoints")
        st.markdown(
            f"""
            **Start:** `({start[0]:.1f}, {start[1]:.1f})`  
            **Goal:** `({goal[0]:.1f}, {goal[1]:.1f})`  
            **Free cells:** `{int(occupancy.size - occupancy.sum()):,}`  
            **Particle population:** `{particle_count}`
            """
        )
        if algorithm == "APF":
            st.caption("Goal attraction and LIDAR-range obstacle repulsion guide each particle.")
        elif algorithm == "Pure Pursuit":
            st.caption("Each particle steers toward a configurable look-ahead point on the Theta* route.")
        else:
            st.caption("The route is shown in the live view; particles independently track it with small steering variations.")
    except ValueError as error:
        st.error(str(error))
        st.stop()

st.divider()
st.subheader("Live particle simulation")
st.caption("Every dot is an independently perturbed point-robot rollout; the dark trace follows the swarm centroid.")
simulation_view = st.empty()
status_view = st.empty()

if run:
    simulation = Simulation(
        grid=occupancy,
        start=start,
        goal=goal,
        algorithm=algorithm,
        particles=particle_count,
        speed=speed,
        timestep=timestep,
        noise=noise,
        lookahead=lookahead,
        lidar_range=lidar_range,
        attraction=attraction,
        repulsion=repulsion,
        goal_tolerance=goal_tolerance,
        seed=int(random_seed),
        planner_options=planner_options,
    )
    height, width = occupancy.shape
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(occupancy, cmap="gray_r", origin="upper", interpolation="nearest")
    route_x, route_y = zip(*simulation.path)
    ax.plot(route_x, route_y, color="#d28a00", linestyle="--", linewidth=1.5, label="Planned route")
    ax.scatter(*start, c="#27d7a1", marker="o", s=80, edgecolors="white", label="Start")
    ax.scatter(*goal, c="#ff5964", marker="*", s=140, edgecolors="white", label="Goal")
    particle_scatter = ax.scatter(
        simulation.positions[:, 0], simulation.positions[:, 1],
        c=np.linspace(0, 1, particle_count), cmap="plasma", s=20, alpha=0.85,
        edgecolors="none", label="Particles",
    )
    centroid_line, = ax.plot([], [], color="#17324d", linewidth=2.2, label="Swarm centroid")
    ax.set(xlim=(-0.5, width - 0.5), ylim=(height - 0.5, -0.5))
    ax.set_aspect("equal")
    ax.legend(loc="upper right", facecolor="white", labelcolor="#172235", fontsize=8)
    ax.set_facecolor("white")
    fig.tight_layout()
    centroid_history: list[np.ndarray] = []
    progress = st.progress(0, text="Starting simulation…")

    for step_number in range(1, max_steps + 1):
        simulation.step()
        if step_number % render_every == 0 or step_number == max_steps or simulation.completion >= 0.95:
            centroid = np.mean(simulation.positions, axis=0)
            centroid_history.append(centroid)
            particle_scatter.set_offsets(simulation.positions)
            if len(centroid_history) > 1:
                history = np.asarray(centroid_history)
                centroid_line.set_data(history[:, 0], history[:, 1])
            simulation_view.pyplot(fig, width="stretch")
            status_view.markdown(
                f"**Step:** {simulation.steps} / {max_steps}　"
                f"**Particles at goal:** {simulation.completion:.0%}　"
                f"**Mean distance to goal:** {simulation.mean_progress:.1f} cells"
            )
            progress.progress(step_number / max_steps, text=f"Simulating · step {step_number}/{max_steps}")
            if playback_delay:
                time.sleep(playback_delay)
        if simulation.completion >= 0.95:
            break

    progress.empty()
    plt.close(fig)
    if simulation.completion >= 0.95:
        st.success(f"Swarm reached the goal region in {simulation.steps} steps.")
    else:
        st.warning(
            f"Simulation reached the {max_steps}-step limit; "
            f"{simulation.completion:.0%} of particles reached the goal region."
        )
    st.session_state["last_simulation_summary"] = {
        "algorithm": algorithm,
        "steps": simulation.steps,
        "completion": simulation.completion,
        "particles": particle_count,
    }
elif "last_simulation_summary" in st.session_state:
    summary = st.session_state["last_simulation_summary"]
    st.info(
        f"Last run: {summary['algorithm']} · {summary['steps']} steps · "
        f"{summary['completion']:.0%} reached · {summary['particles']} particles. "
        "Adjust settings and run again to compare."
    )
else:
    st.info("Choose an algorithm and parameters in the sidebar, then start a live simulation.")

with st.expander("Algorithm references", expanded=False):
    st.markdown(
        """
        Primary papers and technical reports describing the selected planners/controllers:

        - **D* / D* Lite:** Koenig & Likhachev, “D* Lite,” AAAI 2002. [Paper](https://idm-lab.org/bib/abstracts/papers/aaai02b.pdf)
        - **Bi-RRT (RRT-Connect):** Kuffner & LaValle, “RRT-Connect: An Efficient Approach to Single-Query Path Planning,” ICRA 2000. [DOI](https://doi.org/10.1109/ROBOT.2000.844730)
        - **RRT:** LaValle, “Rapidly-Exploring Random Trees: A New Tool for Path Planning,” 1998. [Technical report](https://lavalle.pl/papers/Lav98c.pdf)
        - **Ant colony:** Dorigo, Maniezzo & Colorni, “Ant System: Optimization by a Colony of Cooperating Agents,” IEEE Transactions on Systems, Man, and Cybernetics, 1996. [DOI](https://doi.org/10.1109/3477.484436)
        - **Firefly:** Yang, “Firefly Algorithms for Multimodal Optimization,” SAGA 2009. [DOI](https://doi.org/10.1007/978-3-642-04944-6_14)
        - **PRM:** Kavraki et al., “Probabilistic Roadmaps for Path Planning in High-Dimensional Configuration Spaces,” IEEE Transactions on Robotics and Automation, 1996. [DOI](https://doi.org/10.1109/70.508439)
        - **PRM star:** Karaman & Frazzoli, “Sampling-based Algorithms for Optimal Motion Planning,” The International Journal of Robotics Research, 2011. [DOI](https://doi.org/10.1177/0278364911406761)
        - **A-star:** Hart, Nilsson & Raphael, “A Formal Basis for the Heuristic Determination of Minimum Cost Paths,” IEEE Transactions on Systems Science and Cybernetics, 1968. [DOI](https://doi.org/10.1109/TSSC.1968.300136)
        - **Theta-star:** Daniel et al., “Theta*: Any-Angle Path Planning on Grids,” Journal of Artificial Intelligence Research, 2010. [DOI](https://doi.org/10.1613/jair.2994)
        - **Artificial potential fields (APF):** Khatib, “Real-Time Obstacle Avoidance for Manipulators and Mobile Robots,” The International Journal of Robotics Research, 1986. [DOI](https://doi.org/10.1177/027836498600500106)
        - **Pure Pursuit:** Coulter, “Implementation of the Pure Pursuit Path Tracking Algorithm,” Carnegie Mellon University Robotics Institute, 1992. [Report](https://www.ri.cmu.edu/pub_files/pub3/coulter_r_craig_1992_1/coulter_r_craig_1992_1.pdf)

        Sampling and swarm-based planners are stochastic; repeated runs can produce different routes. PRM star uses a logarithmic nearest-neighbor rule, and D* here uses the D* Lite consistency search on the current occupancy grid.
        """
    )
