from Cython.Build import cythonize
from Cython.Build.BuildExecutable import sysconfig
from setuptools import Extension, setup

python_include_dir = sysconfig.get_path('include')

# Define the extension module
ext_modules = [
    Extension(
        "jumping_moves",
        sources=["jumping_moves.pyx", "king_moves.c", "knight_moves.c", "pawn_moves.c"],
        include_dirs=[python_include_dir, "."],  # Include Python headers and current directory
        extra_compile_args=["-O3", "-march=native", "-funroll-loops", "-flto"],
        extra_link_args=["-O3", "-march=native", "-funroll-loops", "-flto"],
    )
]

# Setup configuration
setup(
    ext_modules=cythonize(ext_modules, compiler_directives={'language_level': "3"}),
    name="jumping_moves")