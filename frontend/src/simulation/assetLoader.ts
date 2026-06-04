export interface MujocoAssetManifest {
  modelRoot: string;
  scene: string;
  files: string[];
}

const ASSET_CACHE_NAME = "pingpong-mujoco-assets-v1";
const ASSET_FETCH_CONCURRENCY = 8;

export async function loadAssetManifest(): Promise<MujocoAssetManifest> {
  const response = await fetch("/assets/mujoco/asset-manifest.json");
  if (!response.ok) {
    throw new Error(`Failed to load MuJoCo asset manifest: ${response.status}`);
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
  const cached = await readCachedAsset(assetPath);
  if (cached) {
    return cached;
  }

  const response = await fetch(assetPath, { cache: "force-cache" });
  if (!response.ok) {
    throw new Error(`Failed to load asset ${assetPath}: ${response.status}`);
  }

  await writeCachedAsset(assetPath, response.clone());
  return new Uint8Array(await response.arrayBuffer());
}

async function readCachedAsset(assetPath: string): Promise<Uint8Array | null> {
  if (!("caches" in window)) {
    return null;
  }

  const cache = await caches.open(ASSET_CACHE_NAME);
  const response = await cache.match(cacheKey(assetPath));
  if (!response || !response.ok) {
    return null;
  }

  return new Uint8Array(await response.arrayBuffer());
}

async function writeCachedAsset(assetPath: string, response: Response): Promise<void> {
  if (!("caches" in window) || !response.ok) {
    return;
  }

  const cache = await caches.open(ASSET_CACHE_NAME);
  await cache.put(cacheKey(assetPath), response);
}

function cacheKey(assetPath: string): string {
  return new URL(assetPath, window.location.origin).toString();
}
