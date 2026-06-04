export interface MujocoAssetManifest {
  modelRoot: string;
  scene: string;
  files: string[];
}

export async function loadAssetManifest(): Promise<MujocoAssetManifest> {
  const response = await fetch("/assets/mujoco/asset-manifest.json");
  if (!response.ok) {
    throw new Error(`Failed to load MuJoCo asset manifest: ${response.status}`);
  }

  return (await response.json()) as MujocoAssetManifest;
}

export async function fetchAssetBytes(assetPath: string): Promise<Uint8Array> {
  const response = await fetch(assetPath);
  if (!response.ok) {
    throw new Error(`Failed to load asset ${assetPath}: ${response.status}`);
  }

  return new Uint8Array(await response.arrayBuffer());
}
