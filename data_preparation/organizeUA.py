import argparse
import os
import sys
import shutil 

CATEGORIES = ['noisereduce','normalized','original']

def make_folders(root:str)->None:
    os.makedirs(root,exist_ok=True)
    for i in CATEGORIES:
        base = os.path.join(root,i)
        os.makedirs(base,exist_ok=True)
    metadata = os.path.join(root,"metadata")
    os.makedirs(metadata,exist_ok=True)
    for i in CATEGORIES:
        os.makedirs(os.path.join(metadata,i),exist_ok=True)

def move_folder(src_folder, dest_folder):
    try:
        shutil.move(src_folder, dest_folder)
        print(f"Moved folder '{src_folder}' to '{dest_folder}' successfully.")
    except Exception as e:
        print(f"Error: {e}")

def copy():
    pass

def move(old,new):
    folders = [os.path.join(old,i) for i in os.listdir(old)]
    for categories in CATEGORIES:
        flag = True
        for folder in folders:
            if categories in folder:
                if flag:
                    src1 = os.path.join(folder,"doc")
                    src2 = os.path.join(folder,"mlf")
                    src3 = os.path.join(folder,"readme_UASpeech.txt")
                    dest1 = os.path.join(new,"metadata",categories,"doc")
                    dest2 = os.path.join(new,"metadata",categories,"mlf")
                    dest3 = os.path.join(new,"metadata","readme_UASpeech.txt")

                    move_folder(src1,dest1)
                    move_folder(src2,dest2)
                    move_folder(src3,dest3)

                    flag = False
            src = os.path.join(folder,"audio",categories)
            if folder[-1] == "C":
                dest = os.path.join(new,categories,"controlled")
            else:
                dest = os.path.join(new,categories,"dysarthria")
            move_folder(src,dest)
            
def main():

    parser = argparse.ArgumentParser(description="Get the path of the input (old) and output folders from the user.")

    parser.add_argument('-d', '--directory', type=str, required=True, help='Path to the old directory')
    parser.add_argument('-o', '--output', type=str, required=True, help='Path to the output directory')
    
    args = parser.parse_args()
    old_folder_path = args.directory
    output_folder_path = args.output
    
    make_folders(output_folder_path)
    move(old_folder_path,output_folder_path)

if __name__ == "__main__":
    main()
