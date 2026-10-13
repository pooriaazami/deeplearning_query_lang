# Command to initialize a new project
import os
import shutil

def init_project(args):
    project_name = args.project_name
    directory = args.directory
    project_path = os.path.join(directory, project_name)

    if os.path.exists(project_path):
        print(f"Error: Project '{project_name}' already exists in {directory}")
        return

    # Create project structure
    os.makedirs(project_path)
    os.makedirs(os.path.join(project_path, 'src'))
    os.makedirs(os.path.join(project_path, 'tests'))
    os.makedirs(os.path.join(project_path, 'examples'))
    os.makedirs(os.path.join(project_path, 'docs'))

    # Create a sample .dl file
    sample_dl = os.path.join(project_path, 'examples', 'sample.dl')
    with open(sample_dl, 'w') as f:
        f.write("""
MODEL sample_model
    LAYER input INPUT(shape=(784,))
    LAYER dense1 DENSE(units=128, activation='relu')
    LAYER dense2 DENSE(units=10, activation='softmax')
    LAYER output OUTPUT(dense2)
END

DATASET mnist
    SOURCE 'mnist.npz'
    PREPROCESSING
        NORMALIZE(min=0, max=1)
    END
    SPLIT random TRAIN=0.8 VALID=0.1 TEST=0.1
END

TRAINING
    OPTIMIZER adam(learning_rate=0.001)
    LOSS sparse_categorical_crossentropy
    METRICS accuracy
    EPOCHS 10
    BATCH_SIZE 32
END
""".strip())

    # Create a README.md
    readme = os.path.join(project_path, 'README.md')
    with open(readme, 'w') as f:
        f.write(f"# {project_name}\n\nA deep learning project created with dlang.\n")

    print(f"Project '{project_name}' created in {project_path}")