import * as THREE from "three";
import type { MainModule, MjData, MjModel } from "@mujoco/mujoco";

type MujocoRuntime = {
  module: MainModule;
  model: MjModel;
  data: MjData;
};

const VISIBLE_GROUP_LIMIT = 3;
const MUJOCO_TEXTURE_ROLE_COUNT = 10;
const MUJOCO_RGB_TEXTURE_ROLE = 1;

export class MujocoModelScene {
  readonly root = new THREE.Group();

  private readonly bodies = new Map<number, THREE.Group>();
  private readonly geometries: THREE.BufferGeometry[] = [];
  private readonly materials: THREE.Material[] = [];
  private readonly textures: THREE.Texture[] = [];

  constructor(private readonly runtime: MujocoRuntime) {
    this.root.name = "MuJoCo Root";
    this.build();
    this.update();
  }

  update(): void {
    const { model, data } = this.runtime;

    for (const [bodyId, body] of this.bodies) {
      setThreePosition(data.xpos, bodyId, body.position);
      setThreeQuaternion(data.xquat, bodyId, body.quaternion);
      body.updateWorldMatrix(false, true);
    }
  }

  dispose(): void {
    this.root.removeFromParent();

    for (const geometry of this.geometries) {
      geometry.dispose();
    }

    for (const material of this.materials) {
      material.dispose();
    }

    for (const texture of this.textures) {
      texture.dispose();
    }

    this.bodies.clear();
    this.geometries.length = 0;
    this.materials.length = 0;
    this.textures.length = 0;
  }

  private build(): void {
    const { module, model } = this.runtime;
    const meshCache = new Map<number, THREE.BufferGeometry>();

    for (let geomId = 0; geomId < model.ngeom; geomId += 1) {
      if (Number(model.geom_group[geomId]) >= VISIBLE_GROUP_LIMIT) {
        continue;
      }

      const bodyId = Number(model.geom_bodyid[geomId]);
      const geomType = Number(model.geom_type[geomId]);
      const size = readVec3(model.geom_size, geomId);
      const geomName = readName(model.names, model.name_geomadr?.[geomId]);
      const geometry = this.createGeometry(module, model, geomId, geomName, geomType, size, meshCache);
      if (!geometry) {
        continue;
      }

      const material = this.createMaterial(model, geomId, geomType, geomName);
      const mesh = new THREE.Mesh(geometry, material);
      mesh.name = geomName;
      const isPlane = geomType === module.mjtGeom.mjGEOM_PLANE.value || geomName === "floor";
      mesh.castShadow = !isPlane;
      mesh.receiveShadow = true;

      setThreePosition(model.geom_pos, geomId, mesh.position);
      setThreeQuaternion(model.geom_quat, geomId, mesh.quaternion);
      if (geomType === module.mjtGeom.mjGEOM_ELLIPSOID.value) {
        mesh.scale.set(size[0], size[2], size[1]);
      }

      this.getBody(bodyId).add(mesh);
    }

    for (const body of this.bodies.values()) {
      this.root.add(body);
    }
  }

  private getBody(bodyId: number): THREE.Group {
    const existing = this.bodies.get(bodyId);
    if (existing) {
      return existing;
    }

    const body = new THREE.Group();
    body.name = readName(this.runtime.model.names, this.runtime.model.name_bodyadr?.[bodyId]);
    this.bodies.set(bodyId, body);
    return body;
  }

  private createGeometry(
    module: MainModule,
    model: MjModel,
    geomId: number,
    geomName: string,
    geomType: number,
    size: [number, number, number],
    meshCache: Map<number, THREE.BufferGeometry>
  ): THREE.BufferGeometry | null {
    if (geomName === "racket_head_back") {
      return null;
    }

    if (geomType === module.mjtGeom.mjGEOM_PLANE.value) {
      return this.registerGeometry(new THREE.PlaneGeometry(100, 100).rotateX(-Math.PI / 2));
    }

    if (geomType === module.mjtGeom.mjGEOM_SPHERE.value) {
      return this.registerGeometry(new THREE.SphereGeometry(size[0], 32, 18));
    }

    if (geomType === module.mjtGeom.mjGEOM_CAPSULE.value) {
      return this.registerGeometry(new THREE.CapsuleGeometry(size[0], size[1] * 2, 12, 24));
    }

    if (geomType === module.mjtGeom.mjGEOM_CYLINDER.value) {
      if (geomName === "racket_rim") {
        return this.registerGeometry(new THREE.TorusGeometry(size[0], size[2] || 0.0045, 12, 96).rotateX(Math.PI / 2));
      }
      return this.registerGeometry(new THREE.CylinderGeometry(size[0], size[0], size[1] * 2, 96));
    }

    if (geomType === module.mjtGeom.mjGEOM_BOX.value) {
      return this.registerGeometry(new THREE.BoxGeometry(size[0] * 2, size[2] * 2, size[1] * 2));
    }

    if (geomType === module.mjtGeom.mjGEOM_ELLIPSOID.value) {
      return this.registerGeometry(new THREE.SphereGeometry(1, 32, 18));
    }

    if (geomType === module.mjtGeom.mjGEOM_MESH.value) {
      const meshId = Number(model.geom_dataid[geomId]);
      const cached = meshCache.get(meshId);
      if (cached) {
        return cached;
      }

      const geometry = this.registerGeometry(createMeshGeometry(model, meshId));
      meshCache.set(meshId, geometry);
      return geometry;
    }

    return null;
  }

  private createMaterial(model: MjModel, geomId: number, geomType: number, geomName: string): THREE.Material {
    const racketMaterial = this.createRacketMaterial(geomName);
    if (racketMaterial) {
      return racketMaterial;
    }

    if (geomType === this.runtime.module.mjtGeom.mjGEOM_PLANE.value || geomName === "floor") {
      const texture = this.createCheckerTexture();
      const material = new THREE.MeshStandardMaterial({
        color: new THREE.Color(0x7aa6c9),
        map: texture,
        metalness: 0,
        roughness: 0.42,
        transparent: true,
        opacity: 0.88,
        depthWrite: false,
        side: THREE.FrontSide
      });
      this.materials.push(material);
      return material;
    }

    const materialId = Number(model.geom_matid[geomId]);
    const color = materialId >= 0 ? readRgba(model.mat_rgba, materialId) : readRgba(model.geom_rgba, geomId);
    const alpha = clamp01(color[3] ?? 1);
    const texture = materialId >= 0 ? this.createTexture(model, materialId) : null;

    const parameters: THREE.MeshStandardMaterialParameters = {
      color: new THREE.Color(color[0], color[1], color[2]),
      metalness: materialId >= 0 ? clamp01(Number(model.mat_metallic?.[materialId] ?? 0)) : 0,
      roughness: materialId >= 0 ? Math.max(0.48, clamp01(Number(model.mat_roughness?.[materialId] ?? 0.62))) : 0.62,
      transparent: alpha < 0.999,
      opacity: alpha,
      side: THREE.FrontSide
    };
    if (texture) {
      parameters.map = texture;
    }

    const material = new THREE.MeshStandardMaterial(parameters);

    this.materials.push(material);
    return material;
  }

  private createRacketMaterial(geomName: string): THREE.Material | null {
    let material: THREE.MeshStandardMaterial | null = null;
    if (geomName === "racket_head") {
      material = new THREE.MeshStandardMaterial({
        color: 0xb91c1c,
        roughness: 0.74,
        metalness: 0,
        side: THREE.FrontSide,
        polygonOffset: true,
        polygonOffsetFactor: -1,
        polygonOffsetUnits: -1
      });
    } else if (geomName === "racket_rim") {
      material = new THREE.MeshStandardMaterial({
        color: 0xd8c89f,
        roughness: 0.52,
        metalness: 0,
        side: THREE.FrontSide
      });
    } else if (geomName === "racket_handle_core") {
      material = new THREE.MeshStandardMaterial({
        color: 0x7c4f24,
        roughness: 0.72,
        metalness: 0,
        side: THREE.FrontSide
      });
    } else if (geomName === "racket_handle_grip") {
      material = new THREE.MeshStandardMaterial({
        color: 0x111111,
        roughness: 0.82,
        metalness: 0,
        side: THREE.FrontSide
      });
    }

    if (!material) {
      return null;
    }

    this.materials.push(material);
    return material;
  }

  private createCheckerTexture(): THREE.Texture {
    const size = 512;
    const cells = 4;
    const cellSize = size / cells;
    const data = new Uint8Array(size * size * 4);

    for (let y = 0; y < size; y += 1) {
      for (let x = 0; x < size; x += 1) {
        const edge = x % cellSize < 3 || y % cellSize < 3;
        const bright = (Math.floor(x / cellSize) + Math.floor(y / cellSize)) % 2 === 0;
        const offset = (y * size + x) * 4;
        const color = edge ? [128, 191, 235] : bright ? [38, 91, 136] : [22, 61, 103];
        data[offset] = color[0];
        data[offset + 1] = color[1];
        data[offset + 2] = color[2];
        data[offset + 3] = 255;
      }
    }

    const texture = new THREE.DataTexture(data, size, size, THREE.RGBAFormat, THREE.UnsignedByteType);
    texture.repeat.set(8, 8);
    texture.wrapS = THREE.RepeatWrapping;
    texture.wrapT = THREE.RepeatWrapping;
    texture.colorSpace = THREE.SRGBColorSpace;
    texture.needsUpdate = true;
    this.textures.push(texture);
    return texture;
  }

  private createTexture(model: MjModel, materialId: number): THREE.Texture | null {
    const textureId = resolveTextureId(model, materialId);
    if (textureId < 0) {
      return null;
    }

    const width = Number(model.tex_width[textureId]);
    const height = Number(model.tex_height[textureId]);
    const channels = Number(model.tex_nchannel[textureId]);
    const offset = Number(model.tex_adr[textureId]);
    if (width <= 0 || height <= 0 || channels <= 0 || offset < 0) {
      return null;
    }

    const textureData = model.tex_data;
    const rgba = new Uint8Array(width * height * 4);
    for (let pixel = 0; pixel < width * height; pixel += 1) {
      const source = offset + pixel * channels;
      const target = pixel * 4;
      rgba[target] = Number(textureData[source]);
      rgba[target + 1] = channels > 1 ? Number(textureData[source + 1]) : rgba[target];
      rgba[target + 2] = channels > 2 ? Number(textureData[source + 2]) : rgba[target];
      rgba[target + 3] = channels > 3 ? Number(textureData[source + 3]) : 255;
    }

    const texture = new THREE.DataTexture(rgba, width, height, THREE.RGBAFormat, THREE.UnsignedByteType);
    texture.repeat.set(Number(model.mat_texrepeat[materialId * 2] ?? 1), Number(model.mat_texrepeat[materialId * 2 + 1] ?? 1));
    texture.wrapS = THREE.RepeatWrapping;
    texture.wrapT = THREE.RepeatWrapping;
    texture.needsUpdate = true;

    this.textures.push(texture);
    return texture;
  }

  private registerGeometry<T extends THREE.BufferGeometry>(geometry: T): T {
    this.geometries.push(geometry);
    return geometry;
  }
}

export function mujocoToThree(position: [number, number, number]): THREE.Vector3 {
  return new THREE.Vector3(position[0], position[2], -position[1]);
}

function createMeshGeometry(model: MjModel, meshId: number): THREE.BufferGeometry {
  const vertexStart = Number(model.mesh_vertadr[meshId]) * 3;
  const vertexEnd = vertexStart + Number(model.mesh_vertnum[meshId]) * 3;
  const sourceVertices = model.mesh_vert.subarray(vertexStart, vertexEnd);
  const vertices = new Float32Array(sourceVertices.length);

  for (let index = 0; index < sourceVertices.length; index += 3) {
    vertices[index] = Number(sourceVertices[index]);
    vertices[index + 1] = Number(sourceVertices[index + 2]);
    vertices[index + 2] = -Number(sourceVertices[index + 1]);
  }

  const faceStart = Number(model.mesh_faceadr[meshId]) * 3;
  const faceEnd = faceStart + Number(model.mesh_facenum[meshId]) * 3;
  const sourceFaces = model.mesh_face.subarray(faceStart, faceEnd);
  const index = maxValue(sourceFaces) > 65535 ? new Uint32Array(sourceFaces) : new Uint16Array(sourceFaces);

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(vertices, 3));
  geometry.setIndex(new THREE.BufferAttribute(index, 1));

  const normalAttribute = createNormalAttribute(model, meshId, vertices.length / 3);
  if (normalAttribute) {
    geometry.setAttribute("normal", normalAttribute);
  } else {
    geometry.computeVertexNormals();
  }

  const uvAttribute = createUvAttribute(model, meshId, vertices.length / 3);
  if (uvAttribute) {
    geometry.setAttribute("uv", uvAttribute);
  }

  geometry.computeBoundingSphere();
  return geometry;
}

function createNormalAttribute(model: MjModel, meshId: number, vertexCount: number): THREE.BufferAttribute | null {
  if (Number(model.mesh_normaladr?.[meshId] ?? -1) < 0 || Number(model.mesh_normalnum?.[meshId] ?? 0) <= 0) {
    return null;
  }

  const normalStart = Number(model.mesh_normaladr[meshId]) * 3;
  const normalEnd = normalStart + Number(model.mesh_normalnum[meshId]) * 3;
  const sourceNormals = model.mesh_normal.subarray(normalStart, normalEnd);
  if (sourceNormals.length < vertexCount * 3) {
    return null;
  }

  const normals = new Float32Array(vertexCount * 3);
  for (let index = 0; index < vertexCount * 3; index += 3) {
    normals[index] = Number(sourceNormals[index]);
    normals[index + 1] = Number(sourceNormals[index + 2]);
    normals[index + 2] = -Number(sourceNormals[index + 1]);
  }

  return new THREE.BufferAttribute(normals, 3);
}

function createUvAttribute(model: MjModel, meshId: number, vertexCount: number): THREE.BufferAttribute | null {
  if (Number(model.mesh_texcoordadr[meshId]) < 0 || Number(model.mesh_texcoordnum[meshId]) <= 0) {
    return null;
  }

  const uvStart = Number(model.mesh_texcoordadr[meshId]) * 2;
  const uvEnd = uvStart + Number(model.mesh_texcoordnum[meshId]) * 2;
  const sourceUv = model.mesh_texcoord.subarray(uvStart, uvEnd);
  const uv = new Float32Array(vertexCount * 2);

  const faceStart = Number(model.mesh_faceadr[meshId]) * 3;
  const faceEnd = faceStart + Number(model.mesh_facenum[meshId]) * 3;
  const faceVertices = model.mesh_face.subarray(faceStart, faceEnd);
  const faceUv = model.mesh_facetexcoord.subarray(faceStart, faceEnd);

  for (let index = 0; index < faceVertices.length; index += 1) {
    const vertexId = Number(faceVertices[index]);
    const uvId = Number(faceUv[index]);
    if (uvId < 0) {
      continue;
    }

    uv[vertexId * 2] = Number(sourceUv[uvId * 2]);
    uv[vertexId * 2 + 1] = Number(sourceUv[uvId * 2 + 1]);
  }

  return new THREE.BufferAttribute(uv, 2);
}

function setThreePosition(buffer: ArrayLike<number>, index: number, target: THREE.Vector3): void {
  target.set(Number(buffer[index * 3]), Number(buffer[index * 3 + 2]), -Number(buffer[index * 3 + 1]));
}

function setThreeQuaternion(buffer: ArrayLike<number>, index: number, target: THREE.Quaternion): void {
  target.set(
    -Number(buffer[index * 4 + 1]),
    -Number(buffer[index * 4 + 3]),
    Number(buffer[index * 4 + 2]),
    -Number(buffer[index * 4])
  );
}

function readVec3(buffer: ArrayLike<number>, index: number): [number, number, number] {
  return [Number(buffer[index * 3]), Number(buffer[index * 3 + 1]), Number(buffer[index * 3 + 2])];
}

function readRgba(buffer: ArrayLike<number>, index: number): [number, number, number, number] {
  return [
    Number(buffer[index * 4] ?? 1),
    Number(buffer[index * 4 + 1] ?? 1),
    Number(buffer[index * 4 + 2] ?? 1),
    Number(buffer[index * 4 + 3] ?? 1)
  ];
}

function readName(namesBuffer: ArrayLike<number>, start: number | undefined): string {
  if (start === undefined || start < 0) {
    return "";
  }

  const bytes =
    namesBuffer instanceof Uint8Array
      ? namesBuffer
      : new Uint8Array(Array.from({ length: namesBuffer.length }, (_, index) => Number(namesBuffer[index])));
  let end = start;
  while (end < bytes.length && bytes[end] !== 0) {
    end += 1;
  }

  return new TextDecoder("utf-8").decode(bytes.subarray(start, end));
}

function resolveTextureId(model: MjModel, materialId: number): number {
  const roleTexture = Number(model.mat_texid?.[materialId * MUJOCO_TEXTURE_ROLE_COUNT + MUJOCO_RGB_TEXTURE_ROLE] ?? -1);
  if (roleTexture >= 0) {
    return roleTexture;
  }

  return Number(model.mat_texid?.[materialId] ?? -1);
}

function maxValue(values: ArrayLike<number>): number {
  let max = 0;
  for (let index = 0; index < values.length; index += 1) {
    max = Math.max(max, Number(values[index]));
  }
  return max;
}

function clamp01(value: number): number {
  return Math.min(1, Math.max(0, value));
}
