# Steps for Pip

Full instruction [here](https://packaging.python.org/en/latest/tutorials/packaging-projects/)

### Prerequisites
- Install: `python -m pip install --upgrade build`
- Install: `python -m pip install --upgrade twine`
- Register at https://pypi.org/account/register/

### Publish release
- Summarize changes in the [changelog](ChangeLog.md).
- Increment version number [here](pycvvdp/vvdp_data/cvvdp_parameters.json) and [here](pyproject.toml).
- [Optional] Remove previous versions `rm dist/*`
- Create the new package: `python -m build`
- [Optional] Upload to the testpypi: `twine upload --repository testpypi dist/*`
- Upload the package: `twine upload dist/*`. 
  - You need to create a token on the pypi.org web page, then enter `__token__` as username and paste the token password as the password. 

