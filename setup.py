from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

extras_require = {
    'update_checker': ['alteryx-open-src-update-checker >= 2.0.0'],
    'prophet': ['prophet-prebuilt == 1.0.2']
}
extras_require['complete'] = sorted(set(sum(extras_require.values(), [])))

setup(
    name='evalml',
    version='0.49.0',
    author='Alteryx, Inc.',
    author_email='open_source_support@alteryx.com',
    description='EvalML is an AutoML library that builds, optimizes, and evaluates machine learning pipelines using domain-specific objective functions.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/alteryx/evalml/',
    python_requires='>=3.8, <4',
    install_requires=open('core-requirements.txt').readlines() + open('requirements.txt').readlines()[1:],
    extras_require=extras_require,
    tests_require=open('test-requirements.txt').readlines(),
    packages=find_packages(),
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'evalml = evalml.__main__:cli'
        ]
    },
    data_files=[('evalml/tests/data', ['evalml/tests/data/churn.csv',
                                       'evalml/tests/data/daily-min-temperatures.csv',
                                       'evalml/tests/data/fraud_transactions.csv.gz',
                                       'evalml/tests/data/tips.csv',
                                       'evalml/tests/data/titanic.csv'])],
)
