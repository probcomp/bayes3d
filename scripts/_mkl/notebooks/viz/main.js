const messaging = require('./lib/messaging.js');
const arrays = require('./lib/arrays.js');
const viz_pb = require('./lib/viz_pb.js');
const THREE = require('three');
import GUI from 'lil-gui'; 
import Stats from 'stats-js'; 
// import * as THREE from 'three';
import * as BufferGeometryUtils from "three/examples/jsm/utils/BufferGeometryUtils";
import {MapControls} from         'three/examples/jsm/controls/MapControls';

console.log("THREE", THREE)
console.log("BufferGeometryUtils", BufferGeometryUtils)
console.log("THREE", THREE.BufferGeometryUtils)
console.log(MapControls)
/*
 *  Need to define that file next to this one.
 */
const handlers = require('./message_handlers.js');


function initialize_interface(shared_data) {
  const scene = new THREE.Scene();
  const width = window.innerWidth;
  const height = window.innerHeight;
  const camera = new THREE.PerspectiveCamera(50, width / height, 0.1, 1000);
  camera.position.x = 1;
  camera.position.y = 1;
  camera.position.z = 5;

  const grid = new THREE.GridHelper(50, 50, 0x888888, 0x555555);

  const origin = new THREE.Vector3(0, 0, 0);
  const axes = [
    new THREE.Line(
      new THREE.BufferGeometry().setFromPoints(
        [origin, new THREE.Vector3(1, 0, 0)]),
      new THREE.LineBasicMaterial({color: 0xff4444, depthTest: false}),
    ),
    new THREE.Line(
      new THREE.BufferGeometry().setFromPoints(
        [origin, new THREE.Vector3(0, 1, 0)]),
      new THREE.LineBasicMaterial({color: 0x44ff44, depthTest: false}),
    ),
    new THREE.Line(
      new THREE.BufferGeometry().setFromPoints(
        [origin, new THREE.Vector3(0, 0, 1)]),
      new THREE.LineBasicMaterial({color: 0x4444ff, depthTest: false}),
    ),
  ];

  const clock = new THREE.Clock();
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  // SHADOWS
  renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap; // default THREE.PCFShadowMap
  renderer.setSize(width, height);

  renderer.setPixelRatio(window.devicePixelRatio);
  document.body.appendChild(renderer.domElement);

  window.addEventListener('resize', on_window_resize, false);
  function on_window_resize(){
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  }

  const motion_controls = new MapControls(camera, renderer.domElement);
  motion_controls.enableDamping = true;
  motion_controls.zoomSpeed = .5;
  motion_controls.panSpeed = 1.;
  motion_controls.mouseButtons = {
    LEFT:  THREE.MOUSE.ROTATE,
    RIGHT: THREE.MOUSE.PAN
  };

  shared_data.animation_controls = {
    animation_frame: 0,
    play: true,
    frame_rate: 24,
  }

  const ui_controls = new GUI({width: window.innerWidth * .25});

  // Animation controls.
  const anim_folder = ui_controls.addFolder('Animation');
  anim_folder.add(shared_data.animation_controls, 'play');
  anim_folder.add(shared_data.animation_controls, 'frame_rate', 0, 100, 1);
  shared_data.animation_frame_controller = anim_folder.add(
    shared_data, 'current_frame', 0, 1, 1).listen();
  shared_data.animation_frame_controller.onChange(
    value => {
      shared_data.start_time = Date.now() - value * 1000;
      shared_data.animation_update_handler(shared_data.current_frame);
    }
  );

  // Performance stats
  const perf_folder = ui_controls.addFolder('Performance');
  const perf_div = document.createElement('div');
  const stats = new Stats();
  stats.dom.style.position = 'static';
  perf_div.appendChild(stats.dom);
  perf_folder.$children.appendChild(perf_div);
  for (const child of stats.dom.children) child.style.display = '';

  // PIP Renderer
  const scene_render_folder = ui_controls.addFolder('Scene renderer');
  const scene_render_div = document.createElement('div');
  scene_render_div.style.border = '1px #888 solid';

  const pip_width = ui_controls.domElement.offsetWidth - 6;
  const pip_height = Math.floor(pip_width * 9. / 16.);
  const pip_renderer = new THREE.WebGLRenderer();
  pip_renderer.shadowMap.enabled = true;
  pip_renderer.shadowMap.type = THREE.PCFSoftShadowMap; // Optional, for softer shadows

  pip_renderer.setSize(pip_width, pip_height);
  pip_renderer.antialias = true;
  pip_renderer.domElement.style.position = 'static';
  pip_renderer.domElement.classList.add('pip');
  pip_renderer.setPixelRatio(window.devicePixelRatio);
  const in_scene_camera = new THREE.PerspectiveCamera(
    50, pip_width / pip_height, 0.1, 1000);
  in_scene_camera.position.x = 2;
  in_scene_camera.position.y = 1;
  in_scene_camera.position.z = 2;
  in_scene_camera.lookAt(0, 0, 0);

  scene_render_div.appendChild(pip_renderer.domElement);
  scene_render_folder.$children.appendChild(scene_render_div);




  const fixed_scene_objs = [grid]
      .concat(axes)
      .concat([new THREE.AmbientLight(0xffffff, 1.)]);
  fixed_scene_objs.forEach(obj => scene.add(obj));

  function animate() {
    requestAnimationFrame(animate);

    const clock_delta = clock.getDelta();
    if (shared_data.animation_controls.play) {
      const new_frame = Math.floor(
          shared_data.animation_controls.frame_rate *
          (Date.now() - shared_data.start_time) / 1000) % shared_data.num_frames
      if (new_frame != shared_data.current_frame) {
        shared_data.current_frame = new_frame;
        shared_data.animation_update_handler(shared_data.current_frame)
      }
      shared_data.mixers.forEach(mixer => mixer.update(clock_delta * shared_data.animation_controls.frame_rate / 30.));
    }

    motion_controls.update(clock_delta);

    renderer.render(scene, camera);
    pip_renderer.render(scene, in_scene_camera);

    stats.update();
  }
  animate();

  const result = {
    scene: scene,
    motion_controls: motion_controls,
    ui_controls: ui_controls,
    fixed_scene_objs: fixed_scene_objs
  };
  return result;
}




function handle_message(rpc_message) {
    console.log("Handling message of type:", rpc_message.getTypeCase())
    switch (rpc_message.getTypeCase()) {

        case viz_pb.Message.TypeCase.PAYLOAD:
            console.log("Handling 'Payload'...")
            var payload = rpc_message.getPayload() 
            handle_payload(payload, ui, shared_data)
            break;

        case viz_pb.Message.TypeCase.PARTICLE_SWARM:
            console.log("Handling 'Particle Swarm'...")
            var data = arrays.pytree_msg_to_js(
                rpc_message.getParticleSwarm().getData());
            handle_particle_swarm(data, ui, shared_data);
            break;
        default:
            console.log("Not sure what to do with this message type")
    }
}

const shared_data = {
  current_frame: 0,
  animation_update_handler:idx=>{},
  start_time: Date.now(),
  mixers: [],
  actions: []
};

const ui = initialize_interface(shared_data);

const mc = messaging.MessagingClient.ConnectNew();
console.log(mc);
mc.add_handler(handle_message);






function handle_particle_swarm(data, ui, shared_data) {
    if (shared_data.data == data) {
      return;
    }
    shared_data.data = data;

    console.log(data);
    const positions = data.position;

    shared_data.num_frames = positions.shape[0]
    shared_data.animation_frame_controller._max = shared_data.num_frames - 1

    ui.scene.clear();
    ui.fixed_scene_objs.forEach(obj => ui.scene.add(obj));

    const material = new THREE.MeshBasicMaterial({color: 0xffff00});

    const duration = 5;
    const mixers = [];
    const actions = [];
    const times = [...Array(shared_data.num_frames).keys()].map(
      x => duration * x / shared_data.num_frames);
    for (var i = 0; i < Math.min(1000, positions.shape[1]); i++) {
      const geometry = new THREE.SphereGeometry(.001, 4, 4);
      const sphere = new THREE.Mesh(geometry, material);
      sphere.position.fromArray(positions.values.subarray(3 * i));
      ui.scene.add(sphere);
      //if (i > 100) continue;

      const sphere_keyframe_track = new THREE.VectorKeyframeTrack(
        '.position',
        times,
        arrays.strided_slice(positions, arrays.ALL, i).values,
        THREE.InterpolateDiscrete);
      const anim_clip = new THREE.AnimationClip(
        /*name = */'sphere_' + String(i),
        /*duration = */duration,
        /*tracks*/[sphere_keyframe_track]);

      mixers.push(new THREE.AnimationMixer(sphere));
      actions.push(mixers.at(-1).clipAction(anim_clip));
      actions.at(-1).play();
    }

    shared_data.mixers = mixers;
    shared_data.actions = actions;
    };




/* * * * * * * * * * * * * * * * * * *
 *  
 *  
 *  
 * * * * * * * * * * * * * * * * * * */

function linear_index(multiIndex, shape) {
    /**
     * This function takes a multi index and a shape and returns the linear index.
     */
    let linearIndex = 0;
    for (let i = 0; i < multiIndex.length; i++) {
        linearIndex += multiIndex[i] * shape.slice(i + 1).reduce((a, b) => a * b, 1);
    }
    return linearIndex;
}

function get_value(arr, ...multiIndex) {
    var i = linear_index(multiIndex, arr.shape)
    return arr.values[i];
}

function handle_gaussians(meta, data, ui, shared_data) {
    console.log("handling gaussians .... ")

    shared_data.data = data;
    shared_data.num_frames = 1
    shared_data.animation_frame_controller._max = shared_data.num_frames - 1



    const meshes = []
    // DRAW MESHES
    for (var i = 0; i < data.vertices.shape[0]; i++) {    
        const vertices = new Float32Array(arrays.strided_slice(data.vertices, i, arrays.ALL, arrays.ALL).values)
        const faces    = new Int32Array(arrays.strided_slice(data.faces, i, arrays.ALL, arrays.ALL).values)
        const color    = new Float32Array(arrays.strided_slice(data.colors,  i, arrays.ALL).values)
        
        var hexColor = (color[0]*255 << 16) | (color[1]*255 << 8) | color[2]*255;

        const material = new THREE.MeshLambertMaterial({
            color: hexColor,
            transparent: true,
            opacity: color[3]
        });


        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
        geometry.setIndex([...faces]);
        geometry.computeVertexNormals()

        const mesh = new THREE.Mesh(geometry, material);
        mesh.castShadow    = true; // Enable casting shadows
        mesh.receiveShadow = true; // Enable receiving shadows

        meshes.push(mesh)
        ui.scene.add(mesh);
    }

    // ui.scene.add(combineMeshesWithColors(meshes))


}

function handle_spheres(meta, data, ui, shared_data) {

    console.log("handling spheres ....", data.centers.shape )


    const geometry = new THREE.SphereGeometry(1.0);

    // const material = new THREE.MeshBasicMaterial({ vertexColors: true });
    const material = new THREE.MeshLambertMaterial({ vertexColors: true });
    

    const instanceCount = data.centers.shape[0]; // Number of instances
    const instancedMesh = new THREE.InstancedMesh(geometry, material, instanceCount);
    instancedMesh.castShadow    = true; // Enable casting shadows
    instancedMesh.receiveShadow = true; // Enable receiving shadows

    const matrix = new THREE.Matrix4();
    const position = new THREE.Vector3();
    const rotation = new THREE.Quaternion(0,0,0,1);
    const scale = new THREE.Vector3(1., 1., 1.);
    const colors = new Float32Array(instanceCount * 3); // RGB for each instance
    // const transparency = new Float32Array(instanceCount)

    for (let i = 0; i < instanceCount; i++) {
        const center = new Float32Array(arrays.strided_slice(data.centers, i, arrays.ALL).values)
        const rgba   = new Float32Array(arrays.strided_slice(data.colors,  i, arrays.ALL).values)


        position.set(center[0], center[1], center[2]);
        scale.set(data.scales.values[i],data.scales.values[i],data.scales.values[i])
        matrix.compose(position, rotation, scale);
        instancedMesh.setMatrixAt(i, matrix);

        const color = new THREE.Color(rgba[0], rgba[1],rgba[2]);
        color.toArray(colors, i * 3);
        // transparency[i] = rgba[3]
    }
    // Step 4: Create a material

    // Apply colors to the InstancedMesh
    instancedMesh.geometry.setAttribute('color', new THREE.InstancedBufferAttribute(colors, 3));
    // instancedMesh.geometry.setAttribute('opacity', new THREE.InstancedBufferAttribute(transparency, 1));

    // console.log("adding mesh to scene ... ")
    

    // Step 5: Add the mesh to the scene
    ui.scene.add(instancedMesh);

    // ui.scene.add(mesh);

}


// function handle_spheres2(transforms4x4_, colors_, ui, shared_data) {

//     const instanceCount = centers_.shape[0]; 
//     const geometry = new THREE.SphereGeometry(1.0);
//     const material = new THREE.MeshLambertMaterial({ vertexColors: true });
//     const instancedMesh = new THREE.InstancedMesh(geometry, material, instanceCount);
//     instancedMesh.castShadow    = true; 
//     instancedMesh.receiveShadow = true; 

//     const matrix   = new THREE.Matrix4();
//     const position = new THREE.Vector3();
//     const rotation = new THREE.Quaternion(0,0,0,1);
//     const scale    = new THREE.Vector3(1., 1., 1.);
//     const colors   = new Float32Array(instanceCount * 3); // RGB for each instance

//     for (let i = 0; i < instanceCount; i++) {
//         const center = new Float32Array(arrays.strided_slice(centers_, i, arrays.ALL).values)
//         const scales = new Float32Array(arrays.strided_slice(scales_,  i, arrays.ALL).values)
//         const cholesky = new Float32Array(arrays.strided_slice(choleskys_,  i, arrays.ALL, arrays.ALL).values)
//         const rgba   = new Float32Array(arrays.strided_slice(colors_,  i, arrays.ALL).values)
//         const color = new THREE.Color(rgba[0], rgba[1], rgba[2]);

//         color.toArray(colors, i * 3);
//         position.set(center[0], center[1], center[2]);
//         // scale.set(scales[0],scales[1],scales[2])
//         // rotation.set(quat[0],quat[1],quat[2],quat[3])

//         // matrix.compose(position, rotation, scale);

//         matrix.fromArray(cholesky)
//         instancedMesh.setMatrixAt(i, matrix);
//     }

//     instancedMesh.geometry.setAttribute('color', new THREE.InstancedBufferAttribute(colors, 3));

//     ui.scene.add(instancedMesh);
// }


function handle_spheres2(transforms4x4, colors, ui, shared_data) {

    const instanceCount = transforms4x4.shape[0]; 
    const geometry = new THREE.SphereGeometry(1.0);
    const material = new THREE.MeshLambertMaterial({ vertexColors: true });
    const instancedMesh = new THREE.InstancedMesh(geometry, material, instanceCount);
    instancedMesh.castShadow    = true; 
    instancedMesh.receiveShadow = true; 

    const instanceMatrix   = new THREE.Matrix4();
    const instanceColors   = new Float32Array(instanceCount * 3); // RGB for each instance

    for (let i = 0; i < instanceCount; i++) {
        const transform4x4 = new Float32Array(arrays.strided_slice(transforms4x4,  i, arrays.ALL, arrays.ALL).values)
        const rgba   = new Float32Array(arrays.strided_slice(colors,  i, arrays.ALL).values)
        const color = new THREE.Color(rgba[0], rgba[1], rgba[2]);

        color.toArray(instanceColors, i * 3);
        instanceMatrix.fromArray(transform4x4)
        instancedMesh.setMatrixAt(i, instanceMatrix);
    }

    instancedMesh.geometry.setAttribute('color', new THREE.InstancedBufferAttribute(instanceColors, 3));

    ui.scene.add(instancedMesh);
}

function handle_gaussians3(transforms4x4, colors, ui, shared_data) {
    console.log("handling gaussians v3 .... ")
    shared_data.num_frames = 1
    shared_data.animation_frame_controller._max = shared_data.num_frames - 1

    const instanceCount = transforms4x4.shape[0]; 
    const geometry = new THREE.SphereGeometry(1.0);
    const material = new THREE.MeshLambertMaterial({ vertexColors: true });
    const instanceMatrix   = new THREE.Matrix4();
    const instanceColors   = new Float32Array(instanceCount * 3); // RGB for each instance
    const instancedMesh = new THREE.InstancedMesh(geometry, material, instanceCount);
    instancedMesh.castShadow    = true; 
    instancedMesh.receiveShadow = true; 
    

    const meshes = []
    for (let i = 0; i < instanceCount; i++) {
        const transform4x4 = new Float32Array(arrays.strided_slice(transforms4x4,  i, arrays.ALL, arrays.ALL).values)
        const rgba = new Float32Array(arrays.strided_slice(colors,  i, arrays.ALL).values)
        const color = new THREE.Color(rgba[0], rgba[1], rgba[2]);

        color.toArray(instanceColors, i * 3);
        instanceMatrix.fromArray(transform4x4)
        instancedMesh.setMatrixAt(i, instanceMatrix);


    }

    instancedMesh.geometry.setAttribute('color', new THREE.InstancedBufferAttribute(instanceColors, 3));

    ui.scene.add(instancedMesh);

}


function handle_gaussians2(transforms4x4, colors, ui, shared_data) {
    console.log("handling gaussians v2 .... ")

    // shared_data.data = data;
    shared_data.num_frames = 1
    shared_data.animation_frame_controller._max = shared_data.num_frames - 1

    const instanceCount = transforms4x4.shape[0]; 
    

    const meshes = []
    for (let i = 0; i < instanceCount; i++) {
        const color    = new Float32Array(arrays.strided_slice(colors,  i, arrays.ALL).values)
        var hexColor   = (color[0]*255 << 16) | (color[1]*255 << 8) | color[2]*255;
        const material = new THREE.MeshLambertMaterial({
            color: hexColor,
            transparent: true,
            opacity: color[3]
        });
        const transform4x4 = new Float32Array(arrays.strided_slice(transforms4x4,  i, arrays.ALL, arrays.ALL).values)
        const matrix       = new THREE.Matrix4();
        matrix.fromArray(transform4x4)
        
        const geometry = new THREE.SphereGeometry(1.0);
        geometry.applyMatrix4(matrix);

        const mesh = new THREE.Mesh(geometry, material);
        mesh.castShadow    = true; // Enable casting shadows
        mesh.receiveShadow = true; // Enable receiving shadows
        meshes.push(mesh);

        ui.scene.add(mesh);
    }

    // ui.scene.add(combineMeshesWithColors(meshes))

}


function handle_dynamic_gaussians(transformsTxNx4x4, colors, ui, shared_data) {
    console.log("handling gaussians v2 .... ")

    // shared_data.data = data;
    const T = transformsTxNx4x4.shape[0];
    const N = transformsTxNx4x4.shape[1];

    shared_data.num_frames = T
    shared_data.animation_frame_controller._max = shared_data.num_frames - 1
    

    const meshes = []
    for (let i = 0; i < N; i++) {
        const color    = new Float32Array(arrays.strided_slice(colors,  i, arrays.ALL).values)
        var hexColor   = (color[0]*255 << 16) | (color[1]*255 << 8) | color[2]*255;
        const material = new THREE.MeshLambertMaterial({
            color: hexColor,
            transparent: true,
            opacity: color[3]
        });
        const transform4x4 = new Float32Array(arrays.strided_slice(transforms4x4,  i, arrays.ALL, arrays.ALL).values)
        const matrix       = new THREE.Matrix4();
        matrix.fromArray(transform4x4)
        
        const geometry = new THREE.SphereGeometry(1.0);
        geometry.applyMatrix4(matrix);

        const mesh = new THREE.Mesh(geometry, material);
        mesh.castShadow    = true; // Enable casting shadows
        mesh.receiveShadow = true; // Enable receiving shadows
        meshes.push(mesh);

        ui.scene.add(mesh);
    }

    // ui.scene.add(combineMeshesWithColors(meshes))

}


function handle_payload(payload, ui, shared_data) {
    var meta = JSON.parse(payload.getJson())
    var data = arrays.pytree_msg_to_js(payload.getData());
    console.log("Metadata:", meta)
    console.log("Type:", meta['type'], meta.type)
    switch(meta['type']) {
        case "setup":
            console.log("case setup")
            // SET UP THE SCENE
            ui.scene.clear();
            ui.fixed_scene_objs.forEach(obj => ui.scene.add(obj));
            ui.scene.background = new THREE.Color( 0xd3d3d3 );
            const dir_light = new THREE.DirectionalLight( 0xffffff, 2 );
            dir_light.position.set( 0,  .1, 0); //default; light shining from top
            dir_light.castShadow = true; // default false
            dir_light.shadow.camera.visible = true;
            ui.scene.add(dir_light);
            const ambientLight = new THREE.AmbientLight(0x404040, 20); // soft white light
            ui.scene.add(ambientLight);
            break

        case "Gaussians":
            console.log("case Gaussians")
            handle_gaussians(meta, data, ui, shared_data)
            break;
        case "Gaussians2":
                console.log("case Gaussians22")
                handle_gaussians2(
                    data.transforms, 
                    data.colors, 
                    ui, shared_data)
                break;
        case "Dynanic Gaussians":
                console.log("case dynamic Gaussians")
                handle_gaussians2(
                    data.transforms, 
                    data.colors, 
                    ui, shared_data)
                break;
        case "Spheres":
            console.log("case Spheres")
            handle_spheres(meta, data, ui, shared_data)
            break;
        default:
            console.log("UKNOWN TYPE")
    }
}


function combineMeshesWithColors(meshes) {
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
