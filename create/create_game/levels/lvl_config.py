from .create_level_file import CreateLevelFile
from .create_game_marker import CreateGameMarker
import os
import os.path as osp
import json
import sys, inspect
from gym.envs.registration import register


def _parse_json_obj(jf, jf_str):
    if 'lvl_type' in jf and jf['lvl_type'] == 'marker':
        super_class = 'CreateGameMarker'
    else:
        super_class = 'CreateLevelFile'

    if 'reward' in jf:
        reward_type = jf['reward']
    else:
        reward_type = 'sparse'

    lvl_name = "CreateLevel" + jf['name'] + '_Det'

    if os.path.isfile(jf_str):
        format_str = '%s = type("%s", (%s,), {"get_json_file": lambda _: "%s", "get_is_rnd": lambda _: %s, "get_reward_type": lambda _: "%s"})'
        run_code_det = f'{lvl_name}= type(' \
                       f'"{lvl_name}",({super_class},),{{' \
                       f'"get_json_file": lambda *args: r"{jf_str}", ' \
                       f'"get_is_rnd": lambda *args: {"False"}, ' \
                       f'"get_reward_type": lambda *args: "{reward_type}" }})'
        # run_code_det = format_str % (lvl_name, lvl_name, super_class, jf_str, "False", reward_type)
        # run_code_det = repr((run_code_det))
        exec(run_code_det)

        rnd_lvl_name = 'CreateLevel' + jf['name']
        run_code_rnd = f'{rnd_lvl_name}= type(' \
                       f'"{rnd_lvl_name}",({super_class},),{{' \
                       f'"get_json_file": lambda *args: r"{jf_str}", ' \
                       f'"get_is_rnd": lambda *args: {"True"}, ' \
                       f'"get_reward_type": lambda *args: "{reward_type}" }})'
        # run_code_rnd = format_str % (rnd_lvl_name, rnd_lvl_name, super_class, jf_str, "True", reward_type)
        # run_code_rnd = repr(run_code_rnd)
        exec(run_code_rnd)

        other_rnd_lvl_name = 'CreateLevel' + jf['name'] + '_Rnd'
        run_code_other = f'{other_rnd_lvl_name}= type(' \
                         f'"{other_rnd_lvl_name}",({super_class},),{{' \
                         f'"get_json_file": lambda *args: r"{jf_str}", ' \
                         f'"get_is_rnd": lambda *args: {"True"}, ' \
                         f'"get_reward_type": lambda *args: "{reward_type}" }})'
        # run_code_other = format_str % (other_rnd_lvl_name, other_rnd_lvl_name, super_class, jf_str, "True", reward_type)
        # run_code_other = repr(run_code_other)
        exec(run_code_other)
    else:
        format_str = '%s = type("%s", (%s,), {"get_json_file": lambda _: "%s", "get_is_rnd": lambda _: %s, "get_reward_type": lambda _: "%s"})'
        run_code_det = f'{lvl_name}= type(' \
                       f'"{lvl_name}",({super_class},),{{' \
                       f'"get_json_file": lambda *args: "{jf_str}", ' \
                       f'"get_is_rnd": lambda *args: {"False"}, ' \
                       f'"get_reward_type": lambda *args: "{reward_type}" }})'
        # run_code_det = format_str % (lvl_name, lvl_name, super_class, jf_str, "False", reward_type)
        # run_code_det = repr((run_code_det))
        exec(run_code_det)

        rnd_lvl_name = 'CreateLevel' + jf['name']
        run_code_rnd = f'{rnd_lvl_name}= type(' \
                       f'"{rnd_lvl_name}",({super_class},),{{' \
                       f'"get_json_file": lambda *args: "{jf_str}", ' \
                       f'"get_is_rnd": lambda *args: {"True"}, ' \
                       f'"get_reward_type": lambda *args: "{reward_type}" }})'
        # run_code_rnd = format_str % (rnd_lvl_name, rnd_lvl_name, super_class, jf_str, "True", reward_type)
        # run_code_rnd = repr(run_code_rnd)
        exec(run_code_rnd)

        other_rnd_lvl_name = 'CreateLevel' + jf['name'] + '_Rnd'
        run_code_other = f'{other_rnd_lvl_name}= type(' \
                         f'"{other_rnd_lvl_name}",({super_class},),{{' \
                         f'"get_json_file": lambda *args: "{jf_str}", ' \
                         f'"get_is_rnd": lambda *args: {"True"}, ' \
                         f'"get_reward_type": lambda *args: "{reward_type}" }})'
        # run_code_other = format_str % (other_rnd_lvl_name, other_rnd_lvl_name, super_class, jf_str, "True", reward_type)
        # run_code_other = repr(run_code_other)
        exec(run_code_other)




    exec('globals()["%s"] = %s' % (lvl_name, lvl_name))
    exec('globals()["%s"] = %s' % (rnd_lvl_name, rnd_lvl_name))
    exec('globals()["%s"] = %s' % (other_rnd_lvl_name, other_rnd_lvl_name))

    base_package_loc = __name__

    register(
        id=lvl_name + '-v0',
        entry_point = base_package_loc + ':' + lvl_name
    )
    register(
        id=rnd_lvl_name + '-v0',
        entry_point = base_package_loc + ':' + rnd_lvl_name
    )
    register(
        id=other_rnd_lvl_name + '-v0',
        entry_point = base_package_loc + ':' + other_rnd_lvl_name
    )

def register_json_folder(json_folder):
    for f in os.listdir(json_folder):
        if not f.endswith('json'):
            continue
        jf_loc = osp.join(json_folder, f)
        with open(jf_loc, 'r') as jf_f:
            jf = json.load(jf_f)

        _parse_json_obj(jf, jf_loc)

def register_json_str(json_str):
    jf = json.loads(json_str)
    _parse_json_obj(jf, json.dumps(jf).replace('"', '\\"'))




def setup_class_lvls(py_loc):
    exec("import %s" % py_loc)
    mod = sys.modules[py_loc]
    for name, obj in inspect.getmembers(mod):
        if inspect.isclass(obj):
            if name.startswith('CreateLevel'):
                register(
                    id=name + '-v0',
                    entry_point=py_loc + ':' + name,
                )


