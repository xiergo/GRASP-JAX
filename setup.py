from setuptools import setup, find_packages
setup(
    name='GRASP-JAX',
    version='1.0.0',
    description='Protein complex prediction with experimental restraints',
    long_description="Integrating Diverse Experimental Information to Assist Protein Complex Structure Prediction",
    long_description_content_type='text/markdown',
    packages=find_packages(),
    install_requires=['py3Dmol','absl-py','biopython',
                      'chex','dm-haiku','dm-tree',
                      'immutabledict','jax','ml-collections',
                      'numpy','pandas','scipy','optax','joblib',
                      'matplotlib', 'scikit-learn', 'tqdm'],
    include_package_data=True
)