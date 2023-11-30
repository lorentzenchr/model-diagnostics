# Development

Applied principles are:

- Trunk-based development 
- Test driven development (TDD) (in spirit)
- Open source, community contributions are warmly welcome
- Continuous integration (CI) automation

## Tools

We use [hatch](https://hatch.pypa.io) as project management tool.
Environments and scripts are defined in the file `pyproject.toml`.

Apart from the default environment are `docs`, `jupyter`, `lint` and `test`
(for a test matrix). Some useful commands are:

- `hatch run cov` or `hatch run pytest` to run tests
- `hatch -e test run pytest` to run test matrix for multiple versions of python and packages
- `hatch run lint:fmt` to apply `black` and `isort`
- `hatch run lint:all` to check linting, typing, style and security
- `hatch run docs:serve` to run a local server under [localhost:8000](localhost:8000)
   for the website supporting live updates, i.e. you can change the docs and see
   updates immediately. Press Control+C to stop.
- `hatch run jupyter:lab` to start a jupyter lab server

## Release Process

See <https://packaging.python.org/en/latest/tutorials/packaging-projects/>,
<https://hatch.pypa.io/latest/build/> and <https://hatch.pypa.io/latest/publish/>.
This will require credentials for <https://test.pypi.org> and <https://pypi.org>.

- Create a new pull request (PR) and set the new `version` in `__about__.py`.
- Merge the PR on github and sync locally
  - `git checkout main`
  - `git pull origin main` (`origin` might be `upstream`)
- Create annotated git tag on branch main for the current commit (not needed for release candidates):
  `git tag -a v1.4.0 -m "Release v1.4.0"`
- Push the annotated tag: `git push origin <tag_name>`.
  Note that `origin` might be to be replaced by `upstream` depending on your setup.
- Run `hatch build`.
- Create a github release with the new tag and upload the build artifacts.
  This triggers to build and deploy the website via github pages.
  (The github action `deploy_docs` can also be triggered manually for convenience.)
- Publish build artifacts on test.pypi: `hatch publish -r test`
- Verify the release on test.pypi (best done in some virtual environment):
  `python3 -m pip install --index-url https://test.pypi.org/simple/ --no-deps model-diagnostics`
- Publish build artifacts on pypi (the real thing!): `hatch publish`