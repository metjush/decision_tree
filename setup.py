from setuptools import setup

setup(name='decision_tree',
			version='0.01',
			description='Practice implementation of a classification decision tree',
			classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2.7',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
      ],
      keywords='classification decision tree machine learning random forest',
			url='https://github.com/metjush/decision_tree',
			author='metjush',
			author_email='metjush@gmail.com',
			license='MIT',
			packages=['decision_tree'],
			install_requires=[
				'numpy',
				'sklearn'
			],
			include_package_data=True,
			zip_safe=False)