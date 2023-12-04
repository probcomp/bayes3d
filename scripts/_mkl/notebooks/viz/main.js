const messaging = require('./lib/messaging.js');
const arrays = require('./lib/arrays.js');
const viz_pb = require('./lib/viz_pb.js');
const THREE = require('three');
import GUI from 'lil-gui'; 
import Stats from 'stats-js'; 
import * as BufferGeometryUtils from "three/examples/jsm/utils/BufferGeometryUtils";
import {MapControls} from         'three/examples/jsm/controls/MapControls';
import { MeshDepthMaterial } from 'three';

// Need to define that file next to this one.
const draw_utils = require('./draw_utils.js');


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
// END OF initialize_interface
// <<<<<<<<<<<<<<<<<<<<<<<<<<<


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
            console.log("Not sure what to do with this message type:", rpc_message.getTypeCase())
    }
}
// END OF handle_message
// <<<<<<<<<<<<<<<<<<<<<

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

    shared_data.animation_update_handler(t => {
        console.log("t:", t)
    })

    switch(meta['type']) {
        
        case "setup":
            console.log("case 'setup'")

            // SET UP THE SCENE
            ui.scene.clear();
            ui.fixed_scene_objs.forEach(obj => ui.scene.add(obj));
            // ui.scene.background = new THREE.Color( 0xd3d3d3 );
            ui.scene.background = new THREE.Color( 0xffffff );
            const dir_light = new THREE.DirectionalLight( 0xffffff, 2 );
            dir_light.position.set( 0,  .1, 0); 
            dir_light.castShadow = true; 
            dir_light.shadow.camera.visible = true;
            const ambientLight = new THREE.AmbientLight(0x404040, 20); // soft white light
            ui.scene.add(dir_light);
            ui.scene.add(ambientLight);
            break


        case "spheres":
            console.log("case 'spheres'")

            var instanced_mesh = draw_utils.create_instanced_sphere_mesh(data.centers, data.colors, data.scales)
            ui.scene.add(instanced_mesh);
            shared_data.num_frames = instanced_mesh.count
            shared_data.animation_frame_controller._max = shared_data.num_frames - 1

            break;


        case "animated spheres":
            console.log("case 'animated spheres'")
            console.log(data.centers.shape)

            var T = data.centers.shape[0]
            var N = data.centers.shape[1]

            var xs = arrays.strided_slice(data.centers, 0, arrays.ALL, arrays.ALL)
            var cs = arrays.strided_slice(data.colors, 0, arrays.ALL, arrays.ALL)
            var ss = arrays.strided_slice(data.scales, 0, arrays.ALL)
            xs.shape = [xs.shape[1], xs.shape[2]]
            cs.shape = [cs.shape[1], cs.shape[2]]
            ss.shape = [ss.shape[1]]

            var instanced_mesh = draw_utils.create_instanced_sphere_mesh(xs, cs, ss)
            ui.scene.add(instanced_mesh);

            console.log("instanced_mesh.count", instanced_mesh.count)

            shared_data.num_frames = T
            shared_data.animation_frame_controller._max = shared_data.num_frames - 1

            shared_data.animation_update_handler = t => {
                const xs = arrays.strided_slice(data.centers, t, arrays.ALL, arrays.ALL)
                const cs = arrays.strided_slice(data.colors,  t, arrays.ALL, arrays.ALL)
                const ss = arrays.strided_slice(data.scales,  t, arrays.ALL)
                xs.shape = [xs.shape[1], xs.shape[2]]
                cs.shape = [cs.shape[1], cs.shape[2]]
                ss.shape = [ss.shape[1]]
                draw_utils.update_instanced_sphere_mesh(instanced_mesh, xs, cs, ss)
            }

            break;

        case "animated gaussians":
            console.log("animated gaussians", data.transforms.shape)
            var T = data.transforms.shape[0]
            var N = data.transforms.shape[1]
            shared_data.num_frames = T
            shared_data.animation_frame_controller._max = shared_data.num_frames - 1

            var t = 0;
            var transforms = arrays.strided_slice(data.transforms, t, arrays.ALL, arrays.ALL, arrays.ALL);
            var colors = arrays.strided_slice(data.colors,  t, arrays.ALL, arrays.ALL);
            var meshes = draw_utils.create_gaussian_meshes(transforms, colors);
            meshes.forEach(mesh => ui.scene.add(mesh));
            
            
            shared_data.animation_update_handler = t => {
                const transforms = arrays.strided_slice(data.transforms, t, arrays.ALL, arrays.ALL, arrays.ALL)
                const colors     = arrays.strided_slice(data.colors,  t, arrays.ALL, arrays.ALL)
                draw_utils.update_gaussian_meshes(meshes, colors, transforms)
            }

            break;

            
        case "gaussians":
            console.log("case 'gaussians'", data.transforms.shape, data.colors.shape)

            
            // var transforms = arrays.strided_slice(data.transforms, 0, arrays.ALL, arrays.ALL, arrays.ALL);
            // var colors = arrays.strided_slice(data.colors,  0, arrays.ALL, arrays.ALL);

            var meshes = draw_utils.create_gaussian_meshes(data.transforms, data.colors)
            meshes.forEach(mesh => ui.scene.add(mesh));

            shared_data.num_frames = meshes.length
            shared_data.animation_frame_controller._max = shared_data.num_frames - 1
            break;


        default:
            console.log("UKNOWN TYPE:", meta['type'])
    }
}
// END OF handle_payload
// <<<<<<<<<<<<<<<<<<<<<


