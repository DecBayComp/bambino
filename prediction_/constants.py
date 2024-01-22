
# Constants
version = 20180409

dim = 2  # number of dimensions (1 or 2)
L = 1.0  # linear size of the system, in um
x_min = -L / 2.0  # in um
x_max = L / 2.0   # in um
gamma_drag = 400.0  # viscous drag, in fN * s / um
t_step = 0.125  # in seconds
N = int(1.0e3)  # int(1.0e6) or int(1.0e5)
progress_update_interval = 100.0
# Integer. How many intermediate smaller steps are made before the next point is saved
internal_steps_number = 100

# Batch parameters
sleep_time = 0.2
logs_folder = "./logs/"
output_folder = ["../output"]
args_file = "./arguments.dat"
args_lock = "./arguments.lock"
lock_timeout = 300
jobs_count_tars = 12   # 132+12
jobs_count_t_bayes = 132
jobs_count_onsager = 1
jobs_count_chloe = 2
manager_script = "job_manager.py"
DETACHED_PROCESS = 0x00000008

CSV_DELIMITER = ';'


class VARIABLE():
    def __init__(self, type_, path,time_end, time_start, activation_time):
        self.time_end = time_end
        self.time_start = time_start
        self.activation_time = activation_time
        self.type = type_
        self.path = path

var_t0 = VARIABLE(type_='t0', path= r'/pasteur/zeus/projets/p02/hecatonchire/tihana/' + 't0' + '/', time_start=0, activation_time=45,
                  time_end=75)

var_t2 = VARIABLE(type_='t2', path= r'/pasteur/zeus/projets/p02/hecatonchire/tihana/' + 't2' + '/', time_start=0, activation_time=60,
                  time_end=90)

var_t15 = VARIABLE(type_='t15',path = r'/pasteur/zeus/projets/p02/hecatonchire/screens/' + 't15' + '/', time_start=0, activation_time=30,
                   time_end=45)

var_t5 = VARIABLE(type_='t5', path = r'/pasteur/zeus/projets/p02/hecatonchire/screens/' + 't5' + '/' , time_start=0, activation_time=45,
                  time_end=60)


class VARIABLE_LABEL():

    def __init__(self, nom, names, feats, labels_, Color):
        self.nom = nom
        self.names = names
        self.feats = feats
        self.labels_ = labels_
        self.Color = Color


label_norm = VARIABLE_LABEL(
    nom='norm', names=[
        't', 'crawl', 'bend', 'stop', 'head retraction', 'back crawl', 'roll',
        'straight_proba', 'straight_and_light_bend_proba', 'bend_proba', 'curl_proba', 'ball_proba',
        'larva_length_smooth_5',
        'larva_length_deriv_smooth_5', 'S_smooth_5', 'S_deriv_smooth_5',
        'eig_smooth_5', 'eig_deriv_smooth_5', 'angle_upper_lower_smooth_5', 'angle_upper_lower_deriv_smooth_5',
        'angle_downer_upper_smooth_5', 'angle_downer_upper_deriv_smooth_5', 'd_eff_head_norm_smooth_5',
        'd_eff_head_norm_deriv_smooth_5', 'd_eff_tail_norm_smooth_5', 'd_eff_tail_norm_deriv_smooth_5',
        'motion_velocity_norm_smooth_5', 'head_velocity_norm_smooth_5', 'tail_velocity_norm_smooth_5',
        'As_smooth_5', 'prod_scal_1', 'prod_scal_2', 'motion_to_u_tail_head_smooth_5',
        'motion_to_v_tail_head_smooth_5'], feats=['t', 'crawl', 'bend', 'stop', 'head retraction', 'back crawl', 'roll', 'As_smooth_5', 'ball_proba'], labels_=['crawl', 'bend', 'stop', 'head retraction', 'back crawl', 'roll'],
    Color=['#17202a', '#c0392b', '#8bc34a', '#2e86c1', '#26c6da', '#f1c40f'])

label_large = VARIABLE_LABEL(
    nom='large', names=[
        't', 'crawl', 'bend', 'stop', 'head retraction', 'back crawl', 'roll', 'small_motion',
        'straight_proba', 'straight_and_light_bend_proba', 'bend_proba', 'curl_proba', 'ball_proba',
        'larva_length_smooth_5',
        'larva_length_deriv_smooth_5', 'S_smooth_5', 'S_deriv_smooth_5',
        'eig_smooth_5', 'eig_deriv_smooth_5', 'angle_upper_lower_smooth_5', 'angle_upper_lower_deriv_smooth_5',
        'angle_downer_upper_smooth_5', 'angle_downer_upper_deriv_smooth_5', 'd_eff_head_norm_smooth_5',
        'd_eff_head_norm_deriv_smooth_5', 'd_eff_tail_norm_smooth_5', 'd_eff_tail_norm_deriv_smooth_5',
        'motion_velocity_norm_smooth_5', 'head_velocity_norm_smooth_5', 'tail_velocity_norm_smooth_5',
        'As_smooth_5', 'prod_scal_1', 'prod_scal_2', 'motion_to_u_tail_head_smooth_5',
        'motion_to_v_tail_head_smooth_5'], feats=['t', 'crawl', 'bend', 'stop', 'head retraction', 'back crawl', 'roll', 'small_motion', 'As_smooth_5', 'ball_proba'], labels_=['crawl', 'bend', 'stop', 'head retraction', 'back crawl', 'roll', 'small_motion'],
    Color=['#17202a', '#c0392b', '#8bc34a', '#2e86c1', '#26c6da', '#f1c40f', '#99a0ab'])

label_weak_strong = VARIABLE_LABEL(nom='weak_strong', names=['t', 'crawl_weak', 'crawl_strong', 'bend_weak', 'bend_strong', 'stop_weak',  'stop_strong', 'head retraction weak', 'head retraction strong',
                                                             'back crawl weak', 'back crawl strong', 'roll weak', 'roll strong', 'straight_proba', 'straight_and _light _bend _proba',  'bend_proba', 'curl_proba', 'ball_proba',
                                                             'larva_length _smooth _ 5', 'larva_length _deriv _smooth _ 5', 'S_smooth _ 5',  'S_deriv _smooth_5',
                                                             'eig_smooth _ 5', 'eig_deriv _smooth _ 5', 'angle_upper _lower _smooth _ 5', 'angle_upper_lower_deriv_smooth_5',
                                                             'angle_downer _upper _smooth _ 5', 'angle_downer _upper _deriv _smooth_5', 'd_eff_head_norm_smooth_5', 'd_eff_head_norm_deriv_smooth_5', 'd_eff_tail_norm_smooth_5',  'd_eff _tail _norm _deriv _smooth _ 5',
                                                             'motion_velocity _norm _smooth _ 5', 'head_velocity _norm _smooth _ 5', 'tail_velocity _norm_smooth _ 5', 'As_smooth_5', 'prod_scal_1', 'prod_scal _ 2', 'motion_to_u_tail_head_smooth_5', 'motion_to_v_tail _head _smooth _ 5'], labels_=['crawl_weak', 'crawl_strong', 'bend_weak', 'bend_strong', 'stop_weak',  'stop_strong', 'head retraction weak', 'head retraction strong',
                                                                                                                                                                                                                                                                                                        'back crawl weak', 'back crawl strong', 'roll weak', 'roll strong'], feats=['t', 'crawl_weak', 'crawl_strong', 'bend_weak', 'bend_strong', 'stop_weak',  'stop_strong', 'head retraction weak', 'head retraction strong',
                                                                                                                                                                                                                                                                                                                                                                                    'back crawl weak', 'back crawl strong', 'roll weak', 'roll strong', 'As_smooth_5', 'ball_proba'],
                                   Color=['#777b81', '#17202a', '#d77167', '#c0392b', '#bce192', '#8bc34a', '#72a6c9', '#2e86c1', '#66ccd8', '#26c6da', '#e1c866', '#f1c40f'])
