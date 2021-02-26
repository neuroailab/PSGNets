import os
import tensorflow as tf
import time
import tqdm
import pdb
import copy


class TrainFramework(object):
    def __init__(self, **params):
        self.params = params
        self.save_params = params['save_params']
        self.load_params = params.get('load_params', None)
        self.train_params = params['train_params']
        self.model_params = params['model_params']
        self.loss_params = params['loss_params']
        self.validation_params = params['validation_params']

        self.val_only = False
        self.profile = False

        # Set cache directory
        self.cache_dir = self.save_params['cache_dir']
        os.system('mkdir -p %s' % self.cache_dir)

        self.log_file_path = os.path.join(self.cache_dir, 'log.txt')
        self.val_log_file_path = os.path.join(self.cache_dir, 'val_log.txt')

        if self.load_params is not None:
            step = self.load_params['query']['step']
            self.load_from_curr_exp = os.path.join(self.cache_dir, 'model.ckpt-%d' % step)
        else:
            self.load_from_curr_exp = tf.train.latest_checkpoint(self.cache_dir)

        if not self.load_from_curr_exp:
            self.log_writer = open(self.log_file_path, 'w')
            self.val_log_writer = open(self.val_log_file_path, 'w')
        else:
            self.log_writer = open(self.log_file_path, 'a+')
            self.val_log_writer = open(self.val_log_file_path, 'a+')

    def build_inputs(self):
        data_params = self.train_params['data_params']
        func = data_params.pop('func')
        self.inputs = func(**data_params)

    def build_network(self, inputs, train):
        model_params = self.model_params
        func = model_params.pop('func')
        outputs, _ = func(
                inputs=inputs,
                train=train,
                **model_params)
        model_params['func'] = func

        if 'trainable_scopes' in model_params:
            trainable_scopes = model_params['trainable_scopes']
            all_train_ref = tf.get_collection_ref(
                    tf.GraphKeys.TRAINABLE_VARIABLES)
            cp_all_train_ref = copy.copy(all_train_ref)
            for each_v in cp_all_train_ref:
                should_be_trainable = False
                for each_trainable_scope in trainable_scopes:
                    if each_v.op.name.startswith(each_trainable_scope):
                        should_be_trainable = True
                if not should_be_trainable:
                    all_train_ref.remove(each_v)
        return outputs

    def build_train_op(self):
        loss_params = self.loss_params

        loss_func_kwargs = loss_params.get('loss_func_kwargs', {})
        input_targets = [self.inputs[key] \
                for key in loss_params['pred_targets']]
        if self.loss_params['labels_to_dict']:
            # put labels in dictionary for certain function signatures
            loss_func_kwargs['labels'] = input_targets
            input_targets = []
        func = loss_params['loss_func']
        self.loss_retval = func(
                self.outputs,
                *input_targets,
                **loss_func_kwargs)
        self.loss_retval = loss_params['agg_func'](
                self.loss_retval,
                **loss_params.get('agg_func_kwargs', {}))

        self.global_step = tf.get_variable(
                'global_step', [],
                dtype=tf.int64, trainable=False,
                initializer=tf.constant_initializer(0))
        lr_rate_params = self.params['learning_rate_params']

        if 'warmup_step' in lr_rate_params:
            warmup_step = lr_rate_params.pop('warmup_step')
            init_lr = lr_rate_params.pop('learning_rate')
            global_steps_int = tf.cast(self.global_step, tf.int32)
            warmup_steps_int = tf.constant(warmup_step, dtype=tf.int32)

            global_steps_float = tf.cast(self.global_step, tf.float32)
            warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)

            warmup_percent_done = global_steps_float / warmup_steps_float
            warmup_learning_rate = init_lr * warmup_percent_done

            is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
            learning_rate = (
                (1.0 - is_warmup) * init_lr + is_warmup * warmup_learning_rate)
            lr_rate_params['learning_rate'] = learning_rate

        func = lr_rate_params.pop('func')
        learning_rate = func(global_step=self.global_step, **lr_rate_params)
        self.learning_rate = learning_rate

        opt_params = self.params['optimizer_params']
        func = opt_params.pop('optimizer')
        opt = func(learning_rate=learning_rate, **opt_params)

        with tf.control_dependencies(
                tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            trainables = tf.get_collection_ref(
                    tf.GraphKeys.TRAINABLE_VARIABLES)
            #for v in trainables:
            #    tf.summary.histogram(v.name, v)
            grads_and_vars = opt.compute_gradients(
                self.loss_retval,
                var_list=trainables)
            self.train_op = opt.apply_gradients(grads_and_vars,
                                                global_step=self.global_step)

    def build_train_targets(self):

        extra_targets_params = self.train_params['targets']
        func = extra_targets_params.pop('func')
        train_targets = func(self.inputs, self.outputs, **extra_targets_params)

        train_targets['train_op'] = self.train_op
        train_targets['loss'] = self.loss_retval
        train_targets['learning_rate'] = self.learning_rate

        self.train_targets = train_targets
        for k, v in train_targets.items():
            if k != 'train_op':
                tf.summary.scalar(k, v)

    def build_sess_and_saver(self):
        gpu_options = tf.GPUOptions(allow_growth=True)
        sess = tf.Session(config=tf.ConfigProto(
                allow_soft_placement=True,
                gpu_options=gpu_options,
                ))
        self.sess = sess
        self.saver = tf.train.Saver(max_to_keep=0)

    def build_tf_summary(self):
        self.merged_summary = tf.summary.merge_all()
        self.train_summary_writer = tf.summary.FileWriter(
            self.cache_dir + '/train', self.sess.graph)
        self.val_summary_writer = tf.summary.FileWriter(
            self.cache_dir + '/val')
        self.train_targets['summary'] = self.merged_summary

    def load_from_ckpt(self, ckpt_path):
        print('Restore from %s' % ckpt_path)
        self.saver.restore(self.sess, ckpt_path)

    def init_and_restore(self):
        init_op_global = tf.global_variables_initializer()
        self.sess.run(init_op_global)
        init_op_local = tf.local_variables_initializer()
        self.sess.run(init_op_local)

        if self.load_from_curr_exp:
            self.load_from_ckpt(self.load_from_curr_exp)
        else:
            split_cache_path = self.cache_dir.split('/')
            if self.load_params:
                split_cache_path[-1] = self.load_params['exp_id']
                split_cache_path[-2] = self.load_params['collname']
                split_cache_path[-3] = self.load_params['dbname']
            load_dir = '/'.join(split_cache_path)
            if self.load_params and self.load_params['query']:
                ckpt_path = os.path.join(
                        load_dir,
                        'model.ckpt-%i' % self.load_params['query']['step'])
            else:
                ckpt_path = tf.train.latest_checkpoint(load_dir)
            if ckpt_path:
                print('Restore from %s' % ckpt_path)
                #self.load_from_ckpt(ckpt_path)
                reader = tf.train.NewCheckpointReader(ckpt_path)
                saved_var_shapes = reader.get_variable_to_shape_map()

                all_vars = tf.global_variables()
                all_var_list = {v.op.name: v for v in all_vars}
                filtered_var_list = {}
                for name, var in all_var_list.items():
                    if name in saved_var_shapes:
                        curr_shape = var.get_shape().as_list()
                        saved_shape = saved_var_shapes[name]
                        if (curr_shape == saved_shape):
                            filtered_var_list[name] = var
                        else:
                            print('Shape mismatch for %s: ' % name \
                                    + str(curr_shape) \
                                    + str(saved_shape))
                _load_saver = tf.train.Saver(var_list=filtered_var_list)
                _load_saver.restore(self.sess, ckpt_path)

    def run_each_validation(self, val_key, curr_step):
        agg_res = None
        num_steps = self.validation_params[val_key]['num_steps']
        for _step in tqdm.trange(num_steps, desc=val_key):
            if self.validation_params[val_key].get('valid_loop', None) is None:
                res = self.sess.run(self.all_val_targets[val_key])
            else:
                res = self.validation_params[val_key]['valid_loop']['func'](
                        self.sess, self.all_val_targets[val_key])
            online_func = self.validation_params[val_key]['online_agg_func']
            agg_res = online_func(agg_res, res, _step)
        agg_func = self.validation_params[val_key]['agg_func']
        val_result = agg_func(agg_res, self.cache_dir, curr_step)
        return val_result

    def run_train_loop(self):
        start_step = self.sess.run(self.global_step)
        train_loop = self.train_params.get('train_loop', None)

        if self.val_only:
            self.run_val(start_step)
            return

        profiler = tf.profiler.Profiler(self.sess.graph)

        for curr_step in range(start_step, int(self.train_params['num_steps']+1)):
            self.start_time = time.time()
            run_meta = tf.RunMetadata()
            if train_loop is None:
                run_opt = None
                if self.profile:
                    run_opt = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                train_res = self.sess.run(self.train_targets,
                                          options=run_opt,
                                          run_metadata=None)
            else:
                train_res = train_loop['func'](self.sess, self.train_targets)

            if self.profile:
                profiler.add_step(curr_step, run_meta)
#                profiler.profile_name_scope(options=(tf.profiler.ProfileOptionBuilder.trainable_variables_parameter()))
                opts = (tf.profiler.ProfileOptionBuilder(
                              tf.profiler.ProfileOptionBuilder.time_and_memory())
                              .with_step(curr_step)
                              .with_timeline_output(os.path.join(self.cache_dir, 'profiler')).build())
                profiler.profile_graph(options=opts)


            duration = time.time() - self.start_time

            if curr_step % 10 == 0:
                self.train_summary_writer.add_summary(train_res['summary'],
                                                      curr_step)

            message = 'Step {} ({:.0f} ms) -- '\
                    .format(curr_step, 1000 * duration)
            rep_msg = ['{}: {:.4f}'.format(k, v) \
                    for k, v in train_res.items()
                    if k != 'train_op' and k != 'summary']
            message += ', '.join(rep_msg)
            print(message)

            if curr_step % self.save_params['cache_filters_freq'] == 0 \
                    and curr_step > 0:
                print('Saving model...')
                self.saver.save(
                        self.sess,
                        os.path.join(
                            self.cache_dir,
                            'model.ckpt'),
                        global_step=curr_step)

            self.log_writer.write(message + '\n')
            if curr_step % self.save_params['save_metrics_freq'] == 0 \
                    and curr_step > 0:
                self.log_writer.close()
                self.log_writer = open(self.log_file_path, 'a+')

            if curr_step % self.save_params['save_valid_freq'] == 0 \
                    and curr_step > 0:
                self.run_val(curr_step)

    def run_val(self, curr_step):
        for each_val_key in self.validation_params:
            val_result = self.run_each_validation(each_val_key, curr_step)
            self.val_log_writer.write(
                    '%s: %s\n' % (each_val_key, str(val_result)))
            print(val_result)
        self.val_log_writer.close()
        self.val_log_writer = open(self.val_log_file_path, 'a+')


    def build_train(self):
        self.build_inputs()
        self.outputs = self.build_network(self.inputs, True)
        self.build_train_op()
        self.build_train_targets()

    def build_val_inputs(self, val_key):
        data_params = self.validation_params[val_key]['data_params']
        func = data_params.pop('func')
        val_inputs = func(**data_params)
        return val_inputs

    def build_val_network(self, val_key, val_inputs):
        with tf.name_scope('validation/' + val_key):
            val_outputs = self.build_network(val_inputs, False)
        return val_outputs

    def build_val_targets(self, val_key, val_inputs, val_outputs):
        target_params = self.validation_params[val_key]['targets']
        func = target_params.pop('func')
        val_targets = func(val_inputs, val_outputs, **target_params)
        return val_targets

    def build_val(self):
        tf.get_variable_scope().reuse_variables()
        self.all_val_targets = {}
        for each_val_key in self.validation_params:
            val_inputs = self.build_val_inputs(each_val_key)
            val_outputs = self.build_val_network(each_val_key, val_inputs)
            val_targets = self.build_val_targets(
                    each_val_key, val_inputs, val_outputs)
            self.all_val_targets[each_val_key] = val_targets

    def train(self):
        self.build_train()
        self.build_val()

        self.build_sess_and_saver()
        self.build_tf_summary()
        self.init_and_restore()

        self.run_train_loop()