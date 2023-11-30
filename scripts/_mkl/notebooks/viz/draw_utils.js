const arrays = require('./lib/arrays.js');
const THREE = require('three');
import GUI from 'lil-gui'; 
import Stats from 'stats-js'; 
import * as BufferGeometryUtils from "three/examples/jsm/utils/BufferGeometryUtils";


export function combine_meshes(meshes) {
    // Create a new empty BufferGeometry
    let combinedGeometry = new THREE.BufferGeometry();

    // This array will hold all the BufferGeometries from the meshes
    let geometries = [];

    meshes.forEach(mesh => {
        // Clone the geometry to avoid modifying the original geometry
        let geometry = mesh.geometry.clone();

        // Ensure the geometry is a BufferGeometry
        if (!(geometry instanceof THREE.BufferGeometry)) {
            geometry = new THREE.BufferGeometry().fromGeometry(geometry);
        }

        // Apply the mesh's position, rotation, and scale to the geometry
        geometry.applyMatrix4(mesh.matrixWorld);

        // Set vertex colors based on the mesh's material color
        if (mesh.material && mesh.material.color) {
            const color = mesh.material.color;
            const colors = [];

            // Assuming each vertex needs a color
            for (let i = 0; i < geometry.attributes.position.count; i++) {
                colors.push(color.r, color.g, color.b);
            }

            geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
        }

        // Push the geometry to our array
        geometries.push(geometry);
    });

    // Merge all geometries into one
    combinedGeometry = BufferGeometryUtils.mergeBufferGeometries(geometries, false);

    // Create a material that supports vertex colors
    const material = new THREE.MeshPhongMaterial({ vertexColors: true });

    // Return a new mesh with the combined geometry and color-supporting material
    return new THREE.Mesh(combinedGeometry, material);
}
// END OF combine_meshes
// <<<<<<<<<<<<<<<<<<<<<


export function create_gaussian_meshes(transforms4x4, colorsRGBA) {
    const t = 0
    const N = transforms4x4.shape[1]; 
    
    const meshes = []
    for (let i = 0; i < N; i++) {
        const color    = new Float32Array(arrays.strided_slice(colorsRGBA, t,  i, arrays.ALL).values)
        var hexColor   = (color[0]*255 << 16) | (color[1]*255 << 8) | color[2]*255;
        const material = new THREE.MeshLambertMaterial({
            color: hexColor,
            transparent: true,
            opacity: color[3]
        });
        const transform4x4 = new Float32Array(arrays.strided_slice(transforms4x4, t, i, arrays.ALL, arrays.ALL).values)
        const matrix       = new THREE.Matrix4();
        matrix.fromArray(transform4x4)
        
        const geometry = new THREE.SphereGeometry(1.0);
        geometry.applyMatrix4(matrix);

        const mesh = new THREE.Mesh(geometry, material);
        mesh.castShadow    = true; // Enable casting shadows
        mesh.receiveShadow = true; // Enable receiving shadows
        meshes.push(mesh);
    }

    return meshes
}
// END OF create_gaussian_meshes
// <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


export function update_gaussian_meshes(meshes, colorsRGBA, transforms4x4) {
    const t = 0
    for (let i = 0; i < meshes.length; i++) {
        // Update color
        const newColor = new Float32Array(arrays.strided_slice(colorsRGBA, t, i, arrays.ALL).values);
        var newHexColor = (newColor[0] * 255 << 16) | (newColor[1] * 255 << 8) | newColor[2] * 255;
        meshes[i].material.color.setHex(newHexColor);
        meshes[i].material.opacity = newColor[3];
        meshes[i].material.needsUpdate = true;

        // Update transform
        meshes[i].matrixAutoUpdate = false; 
        const transform = new Float32Array(arrays.strided_slice(transforms4x4, t, i, arrays.ALL, arrays.ALL).values)
        const matrix    = new THREE.Matrix4();
        matrix.fromArray(transform)
        meshes[i].matrix.set(...matrix.elements)

        // meshes[i].geometry.applyMatrix4(matrix);
        // meshes[i].geometry.matrix = .set(newTransform4x4);
        meshes[i].geometry.needsUpdate = true;
        meshes[i].matrixWorldNeedsUpdate = true;
    }
}
// End of update_gaussian_meshes



export function create_instanced_sphere_mesh(centers, colorsRGBA, scales) {

    const instanceCount = centers.shape[0]; // Number of instances

    const geometry = new THREE.SphereGeometry(1.0, 3,3);
    const instancedGeometry = new THREE.InstancedBufferGeometry().copy(geometry);
    const colorAttribute = new THREE.InstancedBufferAttribute(new Float32Array(colorsRGBA.values), 4);
    colorAttribute.needsUpdate = true;
    instancedGeometry.setAttribute('color', colorAttribute);

    const material = new THREE.MeshLambertMaterial({ vertexColors: true });

    const instancedMesh = new THREE.InstancedMesh(instancedGeometry, material, instanceCount);
    instancedMesh.castShadow    = true; // Enable casting shadows
    instancedMesh.receiveShadow = true; // Enable receiving shadows
    instancedMesh.instanceMatrix.needsUpdate = true;

    const matrix   = new THREE.Matrix4();
    const position = new THREE.Vector3();
    const rotation = new THREE.Quaternion(0,0,0,1);
    const scale    = new THREE.Vector3(1., 1., 1.);
    const colors   = new Float32Array(instanceCount * 3); // RGB for each instance
    

    for (let i = 0; i < instanceCount; i++) {
        const center = new Float32Array(arrays.strided_slice(centers, i, arrays.ALL).values)
        const rgba   = new Float32Array(arrays.strided_slice(colorsRGBA,  i, arrays.ALL).values)
        // const color  = new THREE.Color(rgba[0], rgba[1],rgba[2]);
        position.set(center[0], center[1], center[2]);
        scale.set(scales.values[i], scales.values[i], scales.values[i])

        matrix.compose(position, rotation, scale);
        instancedMesh.setMatrixAt(i, matrix);
    }

    // instancedGeometry.setAttribute('color', colorAttribute);
    
    
    return instancedMesh;
}
// END OF create_instanced_sphere_mesh
// <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

export function update_instanced_sphere_mesh(instanced_mesh, centers, colorsRGBA, scales) {
    const matrix   = new THREE.Matrix4();
    const position = new THREE.Vector3();
    const rotation = new THREE.Quaternion(0,0,0,1);
    const scale    = new THREE.Vector3(1., 1., 1.);

    for (let i = 0; i < instanced_mesh.count; i++) {
        const center = new Float32Array(arrays.strided_slice(centers, i, arrays.ALL).values)
        // position.set(Math.random(), Math.random(), Math.random());
        position.set(center[0], center[1], center[2]);
        scale.set(scales.values[i], scales.values[i], scales.values[i])

        matrix.compose(position, rotation, scale);
        instanced_mesh.setMatrixAt(i, matrix);

    }

    const colorAttribute = new THREE.InstancedBufferAttribute(new Float32Array(colorsRGBA.values), 4);
    instanced_mesh.geometry.setAttribute('color', colorAttribute);
    instanced_mesh.instanceMatrix.needsUpdate = true;
    colorAttribute.needsUpdate = true;
    
    
    return instanced_mesh;
}
// END OF update_instanced_sphere_mesh


export function create_sphere_meshes(centers, colorsRGB, scales) {

    const instanceCount = centers.shape[0]; // Number of instances

    const geometry = new THREE.SphereGeometry(1.0);
    const material = new THREE.MeshLambertMaterial({ vertexColors: true });
    const instancedMesh = new THREE.InstancedMesh(geometry, material, instanceCount);
    instancedMesh.castShadow    = true; // Enable casting shadows
    instancedMesh.receiveShadow = true; // Enable receiving shadows

    const matrix   = new THREE.Matrix4();
    const position = new THREE.Vector3();
    const rotation = new THREE.Quaternion(0,0,0,1);
    const scale    = new THREE.Vector3(1., 1., 1.);
    const colors   = new Float32Array(instanceCount * 3); // RGB for each instance
    // const transparency = new Float32Array(instanceCount)

    for (let i = 0; i < instanceCount; i++) {
        const center = new Float32Array(arrays.strided_slice(centers, i, arrays.ALL).values)
        const rgba   = new Float32Array(arrays.strided_slice(colorsRGB,  i, arrays.ALL).values)
        const color  = new THREE.Color(rgba[0], rgba[1],rgba[2]);
        position.set(center[0], center[1], center[2]);
        scale.set(scales.values[i], scales.values[i], scales.values[i])

        matrix.compose(position, rotation, scale);
        instancedMesh.setMatrixAt(i, matrix);

        color.toArray(colors, i * 3);
    }

    instancedMesh.geometry.setAttribute('color', new THREE.InstancedBufferAttribute(colors, 3));
    
    
    return instancedMesh;
}