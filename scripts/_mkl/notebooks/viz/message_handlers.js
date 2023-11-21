const messaging = require('./lib/messaging.js');
const arrays = require('./lib/arrays.js');
const viz_pb = require('./lib/viz_pb.js');
const THREE = require('three');
import {MapControls} from 'three/examples/jsm/controls/MapControls';
import GUI from 'lil-gui'; 
import Stats from 'stats-js'; 



export function handle_payload(payload, ui, shared_data) {
    var meta = payload.getJson()
    var data = arrays.pytree_msg_to_js(payload.getData());
    console.log("Metadata:", meta)
    console.log("Data", data)

    shared_data.data = data;
    shared_data.num_frames = 1
    shared_data.animation_frame_controller._max = shared_data.num_frames - 1
  
    ui.scene.clear();
    ui.fixed_scene_objs.forEach(obj => ui.scene.add(obj));
  

// // Example Data (Replace with your data)

const vertices = arrays.strided_slice(data.vertices, 0) 
const colors = arrays.strided_slice(data.colors, 0)
print(vertices)



// console.log(vertices)

// // // Colors (RGB format, values between 0 and 1)
// const colors = data.colors

// // // Create BufferGeometry
// const geometry = new THREE.BufferGeometry();
// geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
// geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

// // // Material with vertexColors
// const material = new THREE.MeshBasicMaterial({ vertexColors: true });

// // // Mesh
// const mesh = new THREE.Mesh(geometry, material);
// ui.scene.add(mesh);
    // var geom = new THREE.BufferGeometry(); 
    // var v1 = new THREE.Vector3(0,0,0);
    // var v2 = new THREE.Vector3(0,500,0);
    // var v3 = new THREE.Vector3(0,500,500);

    // geom.vertices.push(v1);
    // geom.vertices.push(v2);
    // geom.vertices.push(v3);
                    
    // geom.faces.push( new THREE.Face3( 0, 1, 2 ) );
    // geom.computeFaceNormals();
                    
    // var object = new THREE.Mesh( geom, new THREE.MeshNormalMaterial() );
                    
    // object.position.z = -100;//move a bit back - size of 500 is a bit big
    // object.rotation.y = -Math.PI * .5;//triangle is pointing in depth, rotate it -90 degrees on Y
                    
    // ui.scene.add(object);

}