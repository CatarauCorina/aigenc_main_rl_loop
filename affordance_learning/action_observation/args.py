import argparse

def str2bool(v):
    return v.lower() == 'true'

def get_args(arg_str=None):
    parser = argparse.ArgumentParser(description='RL')

    parser.add_argument('--fine-tune', action='store_true', default=False)
    parser.add_argument('--conditioned-aux', action='store_true', default=False)
    parser.add_argument('--conditioned-non-linear', type=str2bool, default=False)
    parser.add_argument('--load-best-name', type=str, default=None)

    parser.add_argument('--usage-policy', action='store_true', default=False)
    parser.add_argument('--usage-loss-coef', type=float, default=1.0)
    parser.add_argument('--normalize-embs', action='store_true', default=False)
    parser.add_argument('--only-vis-embs', action='store_true', default=False)
    parser.add_argument('--random-policy', action='store_true', default=False)
    parser.add_argument('--separate-skip', type=str2bool, default=None)
    parser.add_argument('--pac-bayes', action='store_true', default=False)
    parser.add_argument('--pac-bayes-delta', type=float, default=0.1)
    parser.add_argument('--complexity-scale', type=float, default=10.0)

    parser.add_argument('--create-len-filter', type=float, default=None)


    ########################################################
    # Action sampling related
    parser.add_argument('--sample-clusters', type=str2bool, default=None)
    parser.add_argument('--gt-clusters', action='store_true', default=False)
    parser.add_argument('--strict-gt-clusters', action='store_true', default=False)
    parser.add_argument('--n-clusters', type=int, default=10)

    parser.add_argument('--analysis-angle', type=int, default=None)
    parser.add_argument('--analysis-emb', type=float, default=None)
    parser.add_argument('--train-mix-ratio', type=float, default=None)
    ########################################################



    ########################################################
    # Reco related
    parser.add_argument('--reco-n-prods', type=int, default=10000)
    parser.add_argument('--reco-special-fixed-action-set-size', type=int, default=None)
    parser.add_argument('--reco-max-steps', type=int, default=100)
    parser.add_argument('--reco-normalize-beta', type=str2bool, default=True)
    parser.add_argument('--reco-include-mu', type=str2bool, default=False)
    parser.add_argument('--reco-change-omega', type=str2bool, default=True)
    parser.add_argument('--reco-random-product-view', type=str2bool, default=True)
    parser.add_argument('--reco-normal-time-generator', type=str2bool, default=True)
    parser.add_argument('--reco-deterministic', type=str2bool, default=False)
    parser.add_argument('--reco-no-repeat', type=str2bool, default=False)
    parser.add_argument('--reco-prod-dim', type=int, default=16)
    parser.add_argument('--reco-num-flips', type=int, default=0)
    ########################################################


    ########################################################
    # Stack env related
    parser.add_argument('--stack-reward', type=float, default=1.0)
    parser.add_argument('--stack-no-text-render', type=str2bool, default=True)
    parser.add_argument('--stack-mean-height', type=str2bool, default=True)
    parser.add_argument('--stack-dim', type=int, default=1)
    parser.add_argument('--constrain-physics', action='store_true', default=False)
    parser.add_argument('--contacts-off', type=str2bool, default=True)
    parser.add_argument('--double-place-pen', type=float, default=0.0)
    parser.add_argument('--stack-min-steps', type=int, default=2)
    parser.add_argument('--stack-max-steps', type=int, default=300)
    parser.add_argument('--stack-render-high', type=str2bool, default=True)
    ########################################################

    ########################################################
    # Entropy Reward related
    parser.add_argument('--reward-entropy-coef', type=float, default=0.0)
    ########################################################

    parser.add_argument('--gran-factor', type=float, default=1.0)

    parser.add_argument('--distance-effect', action='store_true', default=False)
    parser.add_argument('--distance-sample', action='store_true', default=False)
    parser.add_argument('--only-pos', action='store_true', default=False)


    ########################################################
    # Action set generation args
    parser.add_argument('--action-seg-loc', type=str, default='./data/action_segs')
    parser.add_argument('--action-random-sample', type=str2bool,
            default=True, help='Randomly sample actions or not.')

    parser.add_argument('--action-set-size', type=int, default=None)
    parser.add_argument('--test-action-set-size', type=int, default=None)
    parser.add_argument('--play-data-folder', type=str, default='./method/embedder/data')
    parser.add_argument('--emb-model-folder', type=str, default='./data/embedder/trained_models/')
    parser.add_argument('--create-play-len', type=int, default=5)
    parser.add_argument('--create-play-run-steps', type=int, default=3)
    parser.add_argument('--create-play-colored', action='store_true', default=False)
    parser.add_argument('--create-play-fixed-tool', action='store_true', default=False)
    parser.add_argument('--play-env-name', type=str, default=None)
    parser.add_argument('--image-resolution', type=int, default=256,
        help='If lower image resolution is to be used in play data')
    parser.add_argument('--image-mask', type=str2bool, default=True)

    parser.add_argument('--input-channels', type=int, default=1,
        help='No. of input channels for HTVAE')

    parser.add_argument('--test-split', action='store_true', default=False)
    parser.add_argument('--eval-split', action='store_true', default=False)
    parser.add_argument('--eval-split-ratio', type=float, default=0.5,
            help='Fraction of action set that is eval')

    parser.add_argument('--both-train-test', type=str2bool, default=False)
    parser.add_argument('--fixed-action-set', action='store_true', default=False)

    parser.add_argument('--load-fixed-action-set', action='store_true', default=False,
        help='For nearest neighbor lookup of discrete policy at evaluation')

    parser.add_argument('--num-z', type=int, default=1)

    parser.add_argument('--weight-decay', type=float, default=0.)

    parser.add_argument('--decay-clipping', action='store_true', default=False)


    ########################################################
    # Method specific args
    ########################################################
    parser.add_argument('--latent-dim', type=int, default=1)
    parser.add_argument('--action-proj-dim', type=int, default=1)
    parser.add_argument('--load-only-actor', type=str2bool, default=True)
    parser.add_argument('--sample-k', action='store_true', default=False)
    parser.add_argument('--do-gumbel-softmax', action='store_true', default=False)
    parser.add_argument('--discrete-fixed-variance', type=str2bool, default=False)
    parser.add_argument('--use-batch-norm', type=str2bool, default=False)

    parser.add_argument('--gt-embs', action='store_true', default=False)


    parser.add_argument(
        '--cont-entropy-coef',
        type=float,
        default=1e-1,
        help='scaling continuous entropy coefficient term further (default: 0.1)')

    # Discrete Beta settings
    parser.add_argument('--discrete-beta', type=str2bool, default=False)
    parser.add_argument('--max-std-width', type=float, default=3.0)
    parser.add_argument('--constrained-effects', type=str2bool, default=True)
    parser.add_argument('--bound-effect', action='store_true', default=False)

    parser.add_argument('--emb-margin', type=float, default=1.1)

    parser.add_argument('--nearest-neighbor', action='store_true', default=False)
    parser.add_argument('--combined-dist', action='store_true', default=False)
    parser.add_argument('--combined-add', action='store_true', default=False)

    parser.add_argument('--no-frame-stack', action='store_true', default=False)

    parser.add_argument('--dist-hidden-dim', type=int, default=64)
    parser.add_argument('--dist-linear-action', type=str2bool, default=True)
    parser.add_argument('--dist-non-linear-final', type=str2bool, default=True)

    parser.add_argument('--exp-logprobs', action='store_true', default=False)
    parser.add_argument('--kl-pen', type=float, default=None)
    parser.add_argument('--cat-kl-loss', type=float, default=None)

    parser.add_argument('--reparam', type=str2bool, default=True)
    parser.add_argument('--no-var', action='store_true', default=False)
    parser.add_argument('--z-mag-pen', type=float, default=None)

    # Distance Model specific
    parser.add_argument('--distance-based', action='store_true', default=False)
    parser.add_argument('--cosine-distance', action='store_true', default=False)

    # Gridworld specific
    parser.add_argument('--up-to-option', type=str, default=None)
    parser.add_argument('--no-diag', type=str2bool, default=True)
    parser.add_argument('--option-penalty', type=float, default=0.0)
    parser.add_argument('--grid-flatten', type=str2bool, default=True)
    parser.add_argument('--grid-playing', action='store_true', default=False)
    parser.add_argument('--play-grid-size', type=int, default=80)
    parser.add_argument('--onehot-state', type=str2bool, default=None)

    parser.add_argument('--not-upto', type=str2bool, default=True)
    parser.add_argument('--orig-crossing-env', type=str2bool, default=False)
    parser.add_argument('--max-grid-steps', type=int, default=50)
    parser.add_argument('--grid-subgoal', type=str2bool, default=True)
    parser.add_argument('--grid-fixed-rivers', type=str2bool, default=False)
    parser.add_argument('--grid-safe-wall', type=str2bool, default=True)

    # Video specific
    parser.add_argument('--vid-dir', type=str, default='./data/vids')
    parser.add_argument('--obs-dir', type=str, default='./data/obs')
    parser.add_argument('--should-render-obs', type=str2bool, default=False)
    parser.add_argument('--result-dir', type=str, default='./data/results')
    parser.add_argument('--vid-fps', type=float, default=5.0)
    parser.add_argument('--eval-only', action='store_true', default=False)
    parser.add_argument('--evaluation-mode', type=str2bool, default=False)

    parser.add_argument('--high-render-dim', type=int, default=256, help='Dimension to render evaluation videos at')
    parser.add_argument('--high-render-freq', type=int, default=50)
    parser.add_argument('--no-test-eval', action='store_true', default=False)
    parser.add_argument('--num-render', type=int, default=None)
    parser.add_argument('--num-eval', type=int, default=None)
    parser.add_argument('--render-info-grid', action='store_true', default=False)
    parser.add_argument('--deterministic-policy', action='store_true', default=False)

    parser.add_argument('--debug-render', action='store_true', default=False)
    parser.add_argument('--render-gifs', action='store_true', default=False)
    parser.add_argument('--verbose-eval', action='store_true', default=True)


    # CREATE specific
    parser.add_argument('--half-tools', type=str2bool, default=True)
    parser.add_argument('--half-tool-ratio', type=float, default=0.5)
    parser.add_argument('--marker-reward', type=str, default='reg',
                    help='Type of reward given for the marker ball [reg, dir]')
    parser.add_argument('--create-target-reward', type=float, default=1.0)
    parser.add_argument('--create-sec-goal-reward', type=float, default=2.0)

    parser.add_argument('--run-interval', type=int, default=10)
    parser.add_argument('--render-high-res', action='store_true', default=False)
    parser.add_argument('--render-ball-traces', action='store_true', default=False)
    parser.add_argument('--render-text', action='store_true', default=False)
    parser.add_argument('--render-changed-colors', action='store_true', default=False)

    # Mega render args
    parser.add_argument('--render-mega-res', action='store_true', default=False)
    parser.add_argument('--render-mega-static-res', action='store_true', default=False)
    parser.add_argument('--mega-res-interval', type=int, default=4)
    parser.add_argument('--anti-alias-blur', type=float, default=0.0)

    parser.add_argument('--render-result-figures', action='store_true', default=False)
    parser.add_argument('--render-borders', action='store_true', default=False)

    parser.add_argument('--success-failures', action='store_true', default=False)
    parser.add_argument('--success-only', action='store_true', default=False)

    parser.add_argument('--exp-type', type=str, default=None,
        help='Type of experiment')
    parser.add_argument('--split-type', type=str, default=None,
        help='Type of Splitting for New tools for create game')
    parser.add_argument('--deterministic-split', action='store_true', default=False)

    # Create environment specific
    parser.add_argument('--create-max-num-steps', type=int, default=30,
        help='Max number of steps to take in create game (Earlier default 25)')
    parser.add_argument('--create-permanent-goal', type=str2bool, default=True)
    parser.add_argument('--large-steps', type=int, default=40,
        help='Large steps (simulation gap) for create game (Earlier default 40)')
    parser.add_argument('--skip-actions', type=int, default=1,
        help='No. of actions to skip over for create game')
    parser.add_argument('--play-large-steps', type=int, default=30,
        help='Large steps (simulation gap) for create game play env')
    parser.add_argument('--no-overlap-env', type=str2bool, default=False)
    parser.add_argument('--threshold-overlap', type=str2bool, default=True)
    args = parser.parse_args()
    return args
