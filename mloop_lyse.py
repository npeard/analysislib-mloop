import lyse
import os
import analysislib.common.mloop.mloop_multishot as mloop_multishot

if __name__ == '__main__':
    # Location of config file with respect to this script
    config_file = os.path.join(os.path.dirname(__file__), "mloop_config.toml")
    mloop_multishot.run_singleshot_multishot(config_file)
else:
    # I don't know how we find ourselves here, but it occurs.
    print("mloop_lyse not in main")