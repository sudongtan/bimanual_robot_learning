# SO-101 model — provenance

- `so101.xml`: unmodified copy of
  [openvinotoolkit/physicalai](https://github.com/openvinotoolkit/physicalai)
  `packages/physicalai-mujoco-so101-plugin/urdf/so101/so101.xml` at commit
  `d58921b83a74172752b188449b2ab33e3f8cd603`. Apache 2.0 (`LICENSE`); the
  plugin's own notice is in `NOTICE.physicalai`.
- `assets/*.stl`: from
  [TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100)
  `Simulation/SO101/assets` at commit `5f6d2b876a53a4872e405b991dd925556c9e38a4`. Apache 2.0. Byte-identical
  (sha256) to the Git LFS meshes physicalai references.

The file also contains a demo scene (blocks, target, floor). Our scene attaches
only the `base` body subtree, which excludes those.
