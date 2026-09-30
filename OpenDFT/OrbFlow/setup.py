from setuptools import setup, find_packages

setup(
    name="scdp",
    version="0.1.0",
    description="OrbFlow: equivariant flow matching for orbital coefficient / density prediction",
    packages=find_packages(),
    package_data={
        "scdp.model.equiformer_v3": ["Jd.pt"],
        "scdp.model.scn": ["Jd.pt"],
    },
    include_package_data=True,
)
