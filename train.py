# =========== Import pytorch libs ===========
import torch
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.utils.data as Data
# =========== Import Option modules ===========
from options.train_options import TrainOptions
# =========== Import Dataset modules ===========
import data
# =========== Import Models ===========
from models import create_model
from trainers.kpe_trainer import KPETrainer
# =========== Import Training tools ===========
# =========== Import util modules ===========
from util.visualizer import Visualizer
from util.iter_counter import IterationCounter
from util.util import draw_pose_from_cords
# =========== Import etc... ===========
import sys
from collections import OrderedDict


if __name__ == "__main__":
    # parse options
    opt = TrainOptions().parse()
    # print options to help debugging
    print(' '.join(sys.argv))

    dataloader = data.create_dataloader(opt)

    # create trainer for our model
    trainer = KPETrainer(opt)

    # create tool for counting iterations
    iter_counter = IterationCounter(opt, len(dataloader))
    # create tool for visualization
    visualizer = Visualizer(opt)
    # cuda는 model안에서 바꿔주는걸로
    for epoch in iter_counter.training_epochs():
        iter_counter.record_epoch_start(epoch)
        for i, data_i in enumerate(dataloader, start=iter_counter.epoch_iter):
            iter_counter.record_one_iteration()
            if i % opt.D_steps_per_G == 0:
                trainer.run_generator_one_step(data_i)



            # Visualizations
            if iter_counter.needs_printing():
                losses = trainer.get_latest_losses()
                visualizer.print_current_errors(epoch, iter_counter.epoch_iter,
                                                losses, iter_counter.time_per_iter)
                visualizer.plot_current_errors(losses, iter_counter.total_steps_so_far)

            if iter_counter.needs_displaying() :
                visuals = dict()
                latest_map = trainer.get_latest_maps()
                visuals['source_color_map'] = latest_map['src_color_map']
                visuals['fake_color_map'] = latest_map['fake_color_map']
                visuals['tgt_color_map'] = latest_map['tgt_color_map']
                visualizer.display_current_results(visuals, epoch, iter_counter.total_steps_so_far, iter_counter.epoch_iter, len(dataloader.dataset), opt)

            if iter_counter.needs_saving():
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, iter_counter.total_steps_so_far))
                trainer.save('latest')
                iter_counter.record_current_iter()
        trainer.update_learning_rate(epoch)
        iter_counter.record_epoch_end()

        if epoch % opt.save_epoch_freq == 0 or \
                epoch == iter_counter.total_epochs:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, iter_counter.total_steps_so_far))
            trainer.save('latest')
            trainer.save(epoch)

    print('Training was successfully finished.')



