from setuptools import setup, find_packages
setup(
    name='GRASP-JAX',
    version='1.0.0',
    description='Protein complex prediction with experimental restraints',
    long_description="Integrating Diverse Experimental Information to Assist Protein Complex Structure Prediction",
    long_description_content_type='text/markdown',
    packages=find_packages(),
    install_requires=['py3Dmol','absl-py==1.0.0','biopython==1.79', 'jax==0.4.30', 'jaxlib==0.4.30', 'tensorflow-cpu==2.13.1', 'h5py==3.11.0',
                      'chex==0.1.86','dm-haiku==0.0.12','dm-tree==0.1.8',
                      'immutabledict==2.0.0','ml-collections==0.1.0',
                      'numpy==1.24.3','pandas==2.0.3','scipy==1.11.1',
                      'matplotlib', 'scikit-learn==1.5.2'],
    include_package_data=True
)
