from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = "-e ."


def get_requirements(
    file_path: str,
) -> List[str]:  # file_path:str is a parameter and List[str] is what function returns
    # requirement.txt will contain list of libraries

    """
    This function will return a list of requirements
    """
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

        return requirements


setup(
    name="mlproject",
    version="0.0.1",
    author="Satyam",
    author_email="Satyam.pant@gmail.com",
    packages=find_packages(),
    # install_requires=['pandas','numpy','seaborn']
    install_requires=get_requirements("requirements.txt"),
)
