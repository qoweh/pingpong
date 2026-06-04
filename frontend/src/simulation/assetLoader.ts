export interface MujocoAssetManifest {
  modelRoot: string;
  scene: string;
  files: string[];
  sceneFormat?: "xml" | "mjb";
  sourceScene?: string;
  sourceFiles?: string[];
}

const ASSET_FETCH_CONCURRENCY = 8;

export async function loadAssetManifest(): Promise<MujocoAssetManifest> {
  const response = await fetch("/assets/mujoco/asset-manifest.json");
  if (!response.ok) {
    throw new Error(`Failed to load simulation asset list: ${response.status}`);
  }

  return (await response.json()) as MujocoAssetManifest;
}

export async function loadMujocoAssets(
  files: string[],
  modelRoot: string,
  onAsset: (file: string, bytes: Uint8Array) => void
): Promise<void> {
  let nextIndex = 0;
  const workers = Array.from({ length: Math.min(ASSET_FETCH_CONCURRENCY, files.length) }, async () => {
    while (nextIndex < files.length) {
      const file = files[nextIndex];
      nextIndex += 1;
      const bytes = await fetchAssetBytes(`${modelRoot}/${file}`);
      onAsset(file, bytes);
    }
  });

  await Promise.all(workers);
}

export async function fetchAssetBytes(assetPath: string): Promise<Uint8Array> {
  const response = await fetch(assetPath, { cache: "force-cache" });
  if (!response.ok) {
    throw new Error(`Failed to load simulation asset ${assetPath}: ${response.status}`);
  }

  return new Uint8Array(await response.arrayBuffer());
}
