PLUGIN_VERSION=2.0.0
PLUGIN_ID=model-drift

plugin:
	cat plugin.json|json_pp > /dev/null
	rm -rf dist
	mkdir dist
	zip --exclude "*.pyc" -r dist/dss-plugin-${PLUGIN_ID}-${PLUGIN_VERSION}.zip plugin.json code-env custom-recipes python-lib resource webapps
