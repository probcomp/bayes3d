import kubric as kb
from render_kb_scene import render_kb_scene

def main(): 
    objs = [] 
    apple = {"asset_id": "apple", "render_filename": "assets/sample_objs/ycb_apple/textured.obj", "position": (0, 0, 1), "scale": 10, "quaternion": kb.Quaternion(axis=[1, 0, 0], degrees=90)}
    peach = {"asset_id": "peach", "render_filename": "assets/sample_objs/ycb_peach/textured.obj", "position": (1,0,1), "scale": 10, "quaternion": kb.Quaternion(axis=[1, 0, 0], degrees=90)}
    orange = {"asset_id": "orange", "render_filename": "assets/sample_objs/ycb_orange/textured.obj", "position": (0,1,1), "scale": 10, "quaternion": kb.Quaternion(axis=[1, 0, 0], degrees=90)}
    # knife = {"asset_id": "knife", "render_filename": "assets/sample_objs/ycb_knife/textured.obj", "position": (1,1,0), "scale": 10, "quaternion": kb.Quaternion(axis=[1, 0, 0], degrees=90)}
    # dice = {"asset_id": "dice", "render_filename": "assets/sample_objs/ycb_dice/textured.obj", "position": (1,1,0), "scale": 10, "quaternion": kb.Quaternion(axis=[1, 0, 0], degrees=90)}
    rubik = {"asset_id": "rubik", "render_filename": "assets/sample_objs/ycb_rubik/textured.obj", "position": (1,1,1), "scale": 10, "quaternion": kb.Quaternion(axis=[1, 0, 0], degrees=90)}
    objs.append(apple)
    objs.append(peach)
    objs.append(orange)
    # objs.append(knife)
    # objs.append(dice)
    objs.append(rubik)
    render_kb_scene(objs, "sample_scene", "output", resolution = (512, 512))

if __name__ == "__main__":
    main()