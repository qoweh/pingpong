export interface MujocoAssetManifest {
  modelRoot: string;
  scene: string;
  files: string[];
  sceneFormat?: "xml" | "mjb";
  sourceScene?: string;
  sourceFiles?: string[];
  fallbackModelRoot?: string;
  fallbackScene?: string;
  fallbackSceneFormat?: "xml" | "mjb";
  fallbackFiles?: string[];
}

export interface AssetLoadProgress {
  loaded: number;
  total: number;
  file: string;
}

const ASSET_FETCH_CONCURRENCY = 8;

export async function loadAssetManifest(): Promise<MujocoAssetManifest> {
  const response = await fetch("/assets/mujoco/asset-manifest.json");
  if (!response.ok) {
    throw new Error(`The 3D scene asset list could not be loaded from the server. HTTP ${response.status}.`);
  }

  return (await response.json()) as MujocoAssetManifest;
}

export async function loadMujocoAssets(
  files: string[],
  modelRoot: string,
  onAsset: (file: string, bytes: Uint8Array) => void,
  onProgress?: (progress: AssetLoadProgress) => void
): Promise<void> {
  let nextIndex = 0;
  let completed = 0;
  const total = files.length;
  const workers = Array.from({ length: Math.min(ASSET_FETCH_CONCURRENCY, files.length) }, async () => {
    while (nextIndex < files.length) {
      const file = files[nextIndex];
      nextIndex += 1;
      const bytes = await fetchAssetBytes(`${modelRoot}/${file}`);
      onAsset(file, bytes);
      completed += 1;
      onProgress?.({ loaded: completed, total, file });
    }
  });

  await Promise.all(workers);
}

export async function fetchAssetBytes(assetPath: string): Promise<Uint8Array> {
  const response = await fetch(assetPath, { cache: "force-cache" });
  if (!response.ok) {
    throw new Error(`A 3D scene asset could not be downloaded. ${assetPath} returned HTTP ${response.status}.`);
  }

  return new Uint8Array(await response.arrayBuffer());
}
