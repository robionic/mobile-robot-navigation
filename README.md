# Mobile Robot Navigation Lab

A Python/Streamlit point-robot navigation simulator with an interactive map builder, selectable planning/controller algorithms, and a live population of independently perturbed robot particles.

Inspired by the [Advanced Mobile Robot Navigation Simulation](https://engrprogrammer-shop.fourthwall.com/en-inr/products/advanced-mobile-robot-navigation-simulation-theta-apf-pure-pursuit-lidar-slam). This is an independent point-robot teaching simulator; it does not reproduce the reference project's four-wheel dynamics or full SLAM stack.

## License and third-party materials

The MIT license applies to original code and materials created for this repository. Third-party assets, data, dependencies, and model weights retain their own licenses and notices; the repository's MIT license does not relicense them.

## Features

- Select **Theta\***, **A\***, **D\***, **Bi-RRT**, **RRT**, **Ant colony**, **Firefly**, **PRM**, **PRM\***, **APF**, or **Pure Pursuit** from the UI.
- Use the generated sample binary occupancy map or upload a binary/grayscale image.
- Resize the map to configurable cell dimensions; choose obstacle threshold and image polarity.
- Edit the 2D point table with `x,y,type` rows for `start`, `goal`, and `obstacle` points. Coordinates use a top-left origin and zero-based map cells.
- Configure particle count, simulation duration, playback cadence, speed, timestep, steering noise, look-ahead, LIDAR range, APF attraction/repulsion, goal tolerance, and random seed.
- Configure RRT iteration/step/goal-bias settings, roadmap sampling/neighbors, ant-colony iteration/population/pheromone/heuristic settings, and firefly population/attraction/randomness settings.
- Watch the swarm, planned route, and centroid trace update live.
- Hover over algorithm and parameter controls to see concise definitions and clickable references.

Theta* and A* plan grid routes; D* uses the D* Lite consistency search for the current occupancy map. RRT and Bi-RRT use randomized tree growth; PRM/PRM* build sampled roadmaps. Ant colony and Firefly use swarm-inspired route search/smoothing. Pure Pursuit tracks its planned route; APF combines goal attraction, local LIDAR-range obstacle repulsion, and route guidance. Particle steering variations make it possible to compare multiple plausible rollouts in one run.

The app's **Algorithm references** expander includes primary papers/reports for all listed planners and controllers.

## Run

From this folder, install the dependencies and launch Streamlit:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit. The initial screen uses the included generated sample binary map; the point table starts empty with map-relative start and goal defaults.

## Run tests

```powershell
python -m unittest discover -s tests -v
```
