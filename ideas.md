---
id: ideas_and_future_directions
status: active
read_only: true
description: A scratchpad for ideas, future directions, and longer-term notes that don't fit into the more structured files. This document is maintained by the human owner and must not be edited by agents. It is intended for the owner's personal use and may contain unfiltered thoughts, brainstorming.
---

# Ideas and Future Directions

## Highest Priority

- Decide: can we stick with the current eco-system and dataset?
- Decide: how much time to invest into an updated VIN?
- apply for Aria Gen2 research kit and simulator that was used for generating the ASE dataset.
- clean up data handling.
- move to university workstation.
- figure out how to do multi-step RL.

- update quarto docs (based on current state of the source code, main.typ and slides_4.typ)

## Problems in current implementation

- when we're intreresteed in finer details we must not downsample the pointcloud or mesh! We were downsampling the mesh to 10% of the original faces.
- we're overfitting! tbh I don't know what I meant with this. I think I might have referred to the calibration plot!?

- perspective / philosophy: what would be helpful to me as a human trying do next best view prediction?
- why do we choose azimuth and elevation bounds? wouldn't it be leading to to better generalization if no bounds have higher variances.

---

request acees to ASE simulator to do RL based data collection
- question: why not use mutliple losses (i.e. entropy, various ordinal classification losses) together?

val confusion matrix indicates that it never predicts the lowest two classes!

---

- First predict relative movement, encode new relative pose, then predict look at (shell).
- or idea as described in .agents/tmp/image.png or hesita paper.
- How can we encode trespasses to new areas that will yield really high rri?
- Explicit 3D gaussian state representations instead of voxel fields?

---

## How to move to multi-step, non-myopic RL; how to scale up?

**Problems**:

- We only have the full set of modalities for the offline ego-trajectories.
- Trajectory: (i.e. >= 2 minutes per scene), single snippet / training sample is two seconds long and consists of 20 frames with temporal stride of 10 frames until the next snippet starts.

- So far we only tried to predict the RRI for a single next best view, anchored at the last frame of the snippet / ego-trajectory. If we wanted to do multi-step prediction, we would not have modalities like:
    > **RGB**: We could use something like Gaussian splatting or world models to generate the full modalities for counterfactual poses that are not part of the original ego-trajectory. Weren't there one-shot reconstruction methods in the aria ecosystem: [egocentric_splats](https://github.com/facebookresearch/egocentric_splats)

    > **SLAM PC**: We could emulate their slam algorithm or we simply generate own slam pc for ASE.

- Given that we'll need multi-step trajectories for RL, we need to update the our data_handling to be able to generate and maintian those multi-step trajectories, where we have rri labels, depth renderings, and point clouds from each of the counterfactual views.
- Also, we cannot maintain trajectories for all candidate views, here we shoudl choose a (noisy) top-k of the candidate views (DECIDE: or should we sample them randomly?, i.e. random walk with N_c candidate views each step and K sampled views that we maintain trajectories for?).



**Questions**:

- How can we formulate a multi-step RL task here?
- How can we use our RRI prediction model to implement a ciritic for actor-critic RL - i.e. expected cumulative RRI?
- Can we simply use all available modalities for the ego-trajectory, and compute the embeding of the intermediate counterfactual views based solely on the available modalities - i.e. the GTMesh? SDF, surface normals, (directional, counted) visibility, camera poses, <TODO: what else can we compute for the counterfactual views based on the GTMesh?)>
- How does the TD target look like for multi-step prediction based on the RRI (reward) prediction model?
- What RL approaches / methods exist and are suitable for our task where we don't have the full set of modalities for the counterfactual views?
- How could the MDP look like for our task?
- Can the Critic network use GT information as input during training? I.e. GT OBBs, GT Segmentation Masks
- Generate synthetic offline multi-step trajectories (with same transition dymanics as currently present for single step)

## Scaling up

- Our current oracle RRI pipeline relies on the GTMesh, these are only available for 100 of the 100,000 scenes (4608 snippets, 19.2% were used for training of VinV3)
- We can generate much more than 60 candidate poses per snippet and generate them within broader distribution of candidate view generation parameters.
- So far we haven't really used the GT Obbs, GT depth or GT semantic segmentation masks, these (especially the latter two) could be used to generate the missing modalities for the counterfactual views (i.e. Gaussian splatting, one-shot GS pre-trained model ?).
- Choose different poses in the ego-trajectories as anchor points (instead of always using the last frame ?)


## Modifications to the VIN (RRI prediction) model

### Ablations

- surface reconstruction is an incredibly powerful input modlity - however we should ablate that!
- ablate our modification of CORAL Loss and coral layer.
- ablate the auxiliary huber loss.

### Feature Encodings and Architecture

- we should increase the number of points in the semi dense pointcloud as well as the semidense grid.
- also use pretrained resnet here!
- How can we implement vin v4 as pure transformer? Let projections (with *all* information that correspond to a certain frustum or unit of space attend to each other)?
  - (scene tokens: voxels / splats / entities, candidate query tokens, projection / frustum tokens, history tokens, entity tokens)
- Maybe early fusion into tokens than one QA branch?
- Tiny **pre-trained** cnn for projection features (i.e. small resnet)? Maybe pretrain on OBB detection?
- Let different modalities from the same sample modulate its cosignals.
- Look at network from video analysis - query-centric-attention - i.e. [QCNet](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhou_Query-Centric_Trajectory_Prediction_CVPR_2023_paper.pdf) and [LMFormer](https://arxiv.org/abs/2504.10275)
- do iterative layer refinement - residual learning similar to LMFormer.
- For each point we can encode from which direction they have been observed - similarly to how [hesita](https://arxiv.org/abs/2508.01014) did it. Here we can average over the different observations or we encode them as incidence-angle histogram.


### Objective and Optimization

- How can we combine the imitation learning signal (i.e. computing the RRI for the given candidate views based on the snippet encodings and supervision with oracle rri) with a continous Action policy that aims to predict the best next view based on the same encodings (without encodings of the candidate poses)?
- why not use a higher value for aux huber loss? Does the aux loss even help? Does it only play a role for initial convergence to the mean? -> ablate
- we can do semantic segmentation and bbox prediction as auxiliary tasks.
- perspective: presence of bboxes might indicate presence of high complexity regions.
- When using a continous action policy, predict relative translation before or after the relative rotation and condition the later on the former?
- Transition to "object-centric" views where the RRI of certain objects (with existing GT and pred OBBs) needs to be maximized could offer a greater variety of different counterfactual views (trajectories).
