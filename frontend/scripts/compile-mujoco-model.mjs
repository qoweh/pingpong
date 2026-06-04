import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import loadMujoco from "@mujoco/mujoco";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const frontendRoot = path.resolve(__dirname, "..");
const assetRoot = path.join(frontendRoot, "public", "assets", "mujoco");
const manifestPath = path.join(assetRoot, "asset-manifest.json");
const outputScene = "pingpong_scene.mjb";
const defaultSourceRoot = "/Users/pilt/project-collection/ros2/graduation-prj/pingpong_rl2/assets";
const sourceRoot = path.resolve(process.env.PINGPONG_MUJOCO_SOURCE_ROOT ?? defaultSourceRoot);
const sourceScene = process.env.PINGPONG_MUJOCO_SOURCE_SCENE ?? "scene.xml";

if (!fs.existsSync(path.join(sourceRoot, sourceScene))) {
  throw new Error(
    `MuJoCo source scene is missing: ${path.join(sourceRoot, sourceScene)}. ` +
      "Set PINGPONG_MUJOCO_SOURCE_ROOT if the RL asset directory moved."
  );
}

const wasmPath = path.join(frontendRoot, "node_modules", "@mujoco", "mujoco", "mujoco.wasm");
const mujoco = await loadMujoco({
  locateFile: (file) => (file.endsWith(".wasm") ? wasmPath : file)
});

const vfs = new mujoco.MjVFS();
for (const file of listFiles(sourceRoot)) {
  vfs.addBuffer(file, fs.readFileSync(path.join(sourceRoot, file)));
}

const model = mujoco.MjModel.from_xml_path(sourceScene, vfs);
mujoco.mj_saveModel(model, outputScene, null);
const mjbBytes = mujoco.FS.readFile(outputScene);
fs.writeFileSync(path.join(assetRoot, outputScene), mjbBytes);

const nextManifest = {
  modelRoot: "/assets/mujoco",
  scene: outputScene,
  sceneFormat: "mjb",
  files: [outputScene]
};
fs.writeFileSync(manifestPath, `${JSON.stringify(nextManifest, null, 2)}\n`);

console.log(
  `Compiled ${path.join(sourceRoot, sourceScene)} to ${outputScene} (${mjbBytes.length.toLocaleString()} bytes).`
);

model.delete();
vfs.delete();

function listFiles(root, directory = "") {
  return fs
    .readdirSync(path.join(root, directory), { withFileTypes: true })
    .flatMap((entry) => {
      const relativePath = path.join(directory, entry.name);
      if (entry.isDirectory()) {
        return listFiles(root, relativePath);
      }
      return relativePath.split(path.sep).join(path.posix.sep);
    });
}
