import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Autoformer_EncDec import series_decomp
from layers.Embed import DataEmbedding_wo_pos
from layers.StandardNorm import Normalize
import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import csv

class DFT_series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, top_k=5):
        super(DFT_series_decomp, self).__init__()
        self.top_k = top_k

    def forward(self, x):
        xf = torch.fft.rfft(x)
        freq = abs(xf)
        freq[0] = 0
        top_k_freq, top_list = torch.topk(freq, 5)
        xf[freq <= top_k_freq.min()] = 0
        x_season = torch.fft.irfft(xf)
        x_trend = x - x_season
        return x_season, x_trend


class MultiScaleSeasonMixing(nn.Module):
    """
    Bottom-up mixing season pattern
    """

    def __init__(self, configs):
        super(MultiScaleSeasonMixing, self).__init__()

        self.down_sampling_layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** i),
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                    ),
                    nn.GELU(),
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                    ),

                )
                for i in range(configs.down_sampling_layers)
            ]
        )

    def forward(self, season_list):

        # mixing high->low
        out_high = season_list[0]
        out_low = season_list[1]
        out_season_list = [out_high.permute(0, 2, 1)]

        for i in range(len(season_list) - 1):
            out_low_res = self.down_sampling_layers[i](out_high)
            out_low = out_low + out_low_res
            out_high = out_low
            if i + 2 <= len(season_list) - 1:
                out_low = season_list[i + 2]
            out_season_list.append(out_high.permute(0, 2, 1))

        return out_season_list


class MultiScaleTrendMixing(nn.Module):
    """
    Top-down mixing trend pattern
    """

    def __init__(self, configs):
        super(MultiScaleTrendMixing, self).__init__()

        self.up_sampling_layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                        configs.seq_len // (configs.down_sampling_window ** i),
                    ),
                    nn.GELU(),
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** i),
                        configs.seq_len // (configs.down_sampling_window ** i),
                    ),
                )
                for i in reversed(range(configs.down_sampling_layers))
            ])

    def forward(self, trend_list):

        # mixing low->high
        trend_list_reverse = trend_list.copy()
        trend_list_reverse.reverse()
        out_low = trend_list_reverse[0]
        out_high = trend_list_reverse[1]
        out_trend_list = [out_low.permute(0, 2, 1)]

        for i in range(len(trend_list_reverse) - 1):
            out_high_res = self.up_sampling_layers[i](out_low)
            out_high = out_high + out_high_res
            out_low = out_high
            if i + 2 <= len(trend_list_reverse) - 1:
                out_high = trend_list_reverse[i + 2]
            out_trend_list.append(out_low.permute(0, 2, 1))

        out_trend_list.reverse()
        return out_trend_list


class PastDecomposableMixing(nn.Module):
    def __init__(self, configs):
        super(PastDecomposableMixing, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.down_sampling_window = configs.down_sampling_window

        self.layer_norm = nn.LayerNorm(configs.d_model)
        self.dropout = nn.Dropout(configs.dropout)
        self.channel_independence = configs.channel_independence

        if configs.decomp_method == 'moving_avg':
            self.decompsition = series_decomp(configs.moving_avg)
        elif configs.decomp_method == "dft_decomp":
            self.decompsition = DFT_series_decomp(configs.top_k)
        else:
            raise ValueError('decompsition is error')

        if not configs.channel_independence:
            self.cross_layer = nn.Sequential(
                nn.Linear(in_features=configs.d_model, out_features=configs.d_ff),
                nn.GELU(),
                nn.Linear(in_features=configs.d_ff, out_features=configs.d_model),
            )

        # Mixing season
        self.mixing_multi_scale_season = MultiScaleSeasonMixing(configs)

        # Mxing trend
        self.mixing_multi_scale_trend = MultiScaleTrendMixing(configs)

        self.out_cross_layer = nn.Sequential(
            nn.Linear(in_features=configs.d_model, out_features=configs.d_ff),
            nn.GELU(),
            nn.Linear(in_features=configs.d_ff, out_features=configs.d_model),
        )

    def forward(self, x_list):
        length_list = []

        for x in x_list:
            _, T, _ = x.size()
            length_list.append(T)

        # Decompose to obtain the season and trend
        season_list = []
        trend_list = []
        for x in x_list:
            season, trend = self.decompsition(x)
            if not self.channel_independence:
                season = self.cross_layer(season)
                trend = self.cross_layer(trend)
            season_list.append(season.permute(0, 2, 1))
            trend_list.append(trend.permute(0, 2, 1))
        # bottom-up season mixing
        out_season_list = self.mixing_multi_scale_season(season_list)
        # top-down trend mixing
        # [batch_size/2, seq_len, feature_dim]
        out_trend_list = self.mixing_multi_scale_trend(trend_list)
        # print(f"out_season_list: {out_season_list}")

        out_list = []


        
        for ori, out_season, out_trend, length in zip(x_list, out_season_list, out_trend_list,
                                                      length_list):
            out = out_season + out_trend
            if self.channel_independence:
                out = ori + self.out_cross_layer(out)
            out_list.append(out[:, :length, :])

        # if not self.training:
        #     self._save_trend_data(x_list, out_season_list, out_trend_list,
        #                                               length_list)
        
        return out_list
        
    def _save_trend_data(self, x_list, out_season_list, out_trend_list, length_list):
        """保存趋势数据和相关信息"""
        if not hasattr(self, 'trend_counter'):
            self.trend_counter = 0
            
        # 创建保存目录
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_folder = f'./test_results/trend_analysis_{timestamp}'
        data_folder = os.path.join(base_folder, 'data')
        plot_folder = os.path.join(base_folder, 'plots')
        
        for folder in [data_folder, plot_folder]:
            if not os.path.exists(folder):
                os.makedirs(folder, exist_ok=True)

        # 从输入中提取数据并转换为numpy数组
        x_data = [x.detach().cpu().numpy() for x in x_list]
        season_data = [s.detach().cpu().numpy() for s in out_season_list]
        trend_data = [t.detach().cpu().numpy() for t in out_trend_list]

        
        # 保存为CSV格式
        for i in range(len(x_data)):
            # 获取当前样本的所有数据长度
            x_len = len(x_data[i].flatten())
            print(f"season_data.shape: {season_data[i].shape}")
            print(f"trend_data.shape: {trend_data[i].shape}")
            season_len = len(season_data[i].flatten())
            trend_len = len(trend_data[i].flatten())
            print(f"x_len: {x_len}, season_len: {season_len}, trend_len: {trend_len}")
            # assert False
            
            # 使用最大长度进行填充
            max_len = max(x_len, season_len, trend_len)
            
            data_dict = {
                'timestep': np.arange(max_len),
                'input': np.pad(x_data[i].flatten(), 
                              (0, max_len - x_len), 
                              mode='constant', 
                              constant_values=np.nan),
                'seasonal': np.pad(season_data[i].flatten(), 
                                 (0, max_len - season_len), 
                                 mode='constant', 
                                 constant_values=np.nan),
                'trend': np.pad(trend_data[i].flatten(), 
                              (0, max_len - trend_len), 
                              mode='constant', 
                              constant_values=np.nan),
            }
            
            if length_list:
                data_dict['length'] = length_list[i]

            df = pd.DataFrame(data_dict)
            csv_path = os.path.join(data_folder, f'decomposition_data_{self.trend_counter}_sample_{i}.csv')
            df.to_csv(csv_path, index=False)

        # 保存平均值（使用相同的长度处理）
        max_len = max(len(x.flatten()) for x in x_data)
        
        mean_dict = {
            'timestep': np.arange(max_len),
            'avg_input': np.mean([np.pad(x.flatten(), 
                                       (0, max_len - len(x.flatten())), 
                                       mode='constant', 
                                       constant_values=np.nan) 
                                for x in x_data], axis=0),
            'avg_seasonal': np.mean([np.pad(s.flatten(), 
                                          (0, max_len - len(s.flatten())), 
                                          mode='constant', 
                                          constant_values=np.nan) 
                                   for s in season_data], axis=0),
            'avg_trend': np.mean([np.pad(t.flatten(), 
                                       (0, max_len - len(t.flatten())), 
                                       mode='constant', 
                                       constant_values=np.nan) 
                                for t in trend_data], axis=0),
        }

        df_mean = pd.DataFrame(mean_dict)
        csv_path = os.path.join(data_folder, f'decomposition_data_{self.trend_counter}_average.csv')
        df_mean.to_csv(csv_path, index=False)

        # 可视化
        # self._visualize_decomposition(x_data, season_data, trend_data, self.trend_counter, plot_folder)
        
        self.trend_counter += 1

    def _visualize_decomposition(self, x_data, season_data, trend_data, counter, save_folder):
        """可视化分解结果"""
        def process_data(data_list):
            # 打印每个数组的形状
            print("Array shapes in list:", [arr.shape for arr in data_list])
            
            # 只使用第一个数组
            data = data_list[0]
            # 转置为 (batch_size, seq_len, feature_dim)
            return np.transpose(data, (2, 1, 0))
        
        try:
            # 打印原始数据信息
            print(f"Number of arrays - x: {len(x_data)}, season: {len(season_data)}, trend: {len(trend_data)}")
            print(f"First array shapes - x: {x_data[0].shape}, season: {season_data[0].shape}, trend: {trend_data[0].shape}")
            
            # 处理数据
            x_processed = process_data(x_data)      # shape: (batch_size, seq_len, feature_dim)
            season_processed = process_data(season_data)
            trend_processed = process_data(trend_data)
            
            # 对特征维度取平均
            x_processed = np.mean(x_processed, axis=2)      # shape: (batch_size, seq_len)
            season_processed = np.mean(season_processed, axis=2)
            trend_processed = np.mean(trend_processed, axis=2)
            
            # 打印处理后的形状
            print(f"Processed shapes - x: {x_processed.shape}, season: {season_processed.shape}, trend: {trend_processed.shape}")
            
            # 为了避免图太多，我们只可视化前几个样本
            num_samples_to_plot = min(5, x_processed.shape[0])
            
            # 单个样本的可视化
            for i in range(num_samples_to_plot):
                fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
                
                # 获取当前样本的数据
                x = x_processed[i]
                seasonal = season_processed[i]
                trend = trend_processed[i]
                
                # 设置时间轴
                time_steps = np.arange(len(x))
                
                # 原始数据
                ax1.plot(time_steps, x, color='#2878B5', linewidth=2, label='Original')
                ax1.set_title(f'Original Time Series (Sample {i+1})', fontsize=12, pad=10)
                ax1.set_ylabel('Value', fontsize=10)
                ax1.grid(True, linestyle='--', alpha=0.7)
                ax1.legend()
                
                # 趋势成分
                ax2.plot(time_steps, trend, color='#C82423', linewidth=2, label='Trend')
                ax2.set_title('Trend Component', fontsize=12, pad=10)
                ax2.set_ylabel('Value', fontsize=10)
                ax2.grid(True, linestyle='--', alpha=0.7)
                ax2.legend()
                
                # 季节性成分
                ax3.plot(time_steps, seasonal, color='#009977', linewidth=2, label='Seasonal')
                ax3.set_title('Seasonal Component', fontsize=12, pad=10)
                ax3.set_xlabel('Time Steps', fontsize=10)
                ax3.set_ylabel('Value', fontsize=10)
                ax3.grid(True, linestyle='--', alpha=0.7)
                ax3.legend()
                
                plt.tight_layout(pad=3.0)
                fig.suptitle(f'Time Series Decomposition - Batch Sample {i+1}', fontsize=14, y=1.02)
                plt.savefig(os.path.join(save_folder, f'decomposition_viz_{counter}_sample_{i+1}.pdf'),
                        bbox_inches='tight', dpi=300)
                plt.close()
            
            # 批次平均值图
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
            
            # 计算批次平均值
            avg_x = np.mean(x_processed, axis=0)
            avg_trend = np.mean(trend_processed, axis=0)
            avg_season = np.mean(season_processed, axis=0)
            
            time_steps = np.arange(len(avg_x))
            
            # 绘制平均值
            ax1.plot(time_steps, avg_x, color='#2878B5', linewidth=2, label='Batch Average')
            ax1.set_title('Average Original Time Series', fontsize=12, pad=10)
            ax1.set_ylabel('Value', fontsize=10)
            ax1.grid(True, linestyle='--', alpha=0.7)
            ax1.legend()
            
            ax2.plot(time_steps, avg_trend, color='#C82423', linewidth=2, label='Batch Average')
            ax2.set_title('Average Trend Component', fontsize=12, pad=10)
            ax2.set_ylabel('Value', fontsize=10)
            ax2.grid(True, linestyle='--', alpha=0.7)
            ax2.legend()
            
            ax3.plot(time_steps, avg_season, color='#009977', linewidth=2, label='Batch Average')
            ax3.set_title('Average Seasonal Component', fontsize=12, pad=10)
            ax3.set_xlabel('Time Steps', fontsize=10)
            ax3.set_ylabel('Value', fontsize=10)
            ax3.grid(True, linestyle='--', alpha=0.7)
            ax3.legend()
            
            plt.tight_layout(pad=3.0)
            fig.suptitle('Average Time Series Decomposition (Across Batch)', fontsize=14, y=1.02)
            plt.savefig(os.path.join(save_folder, f'decomposition_viz_{counter}_batch_average.pdf'),
                    bbox_inches='tight', dpi=300)
            plt.close()
            
        except Exception as e:
            print(f"Error in visualization: {str(e)}")
            print(f"Input types - x: {type(x_data)}, season: {type(season_data)}, trend: {type(trend_data)}")
            if isinstance(x_data, list):
                print(f"First array in list - x: {x_data[0].shape}")
                print("All shapes in x_data:", [x.shape for x in x_data])
            raise e

class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.down_sampling_window = configs.down_sampling_window
        self.channel_independence = configs.channel_independence
        self.pdm_blocks = nn.ModuleList([PastDecomposableMixing(configs)
                                         for _ in range(configs.e_layers)])

        self.preprocess = series_decomp(configs.moving_avg)
        self.enc_in = configs.enc_in

        if self.channel_independence:
            self.enc_embedding = DataEmbedding_wo_pos(1, configs.d_model, configs.embed, configs.freq,
                                                      configs.dropout)
        else:
            self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                      configs.dropout)

        self.layer = configs.e_layers

        self.normalize_layers = torch.nn.ModuleList(
            [
                Normalize(self.configs.enc_in, affine=True, non_norm=True if configs.use_norm == 0 else False)
                for i in range(configs.down_sampling_layers + 1)
            ]
        )

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.predict_layers = torch.nn.ModuleList(
                [
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** i),
                        configs.pred_len,
                    )
                    for i in range(configs.down_sampling_layers + 1)
                ]
            )

            if self.channel_independence:
                self.projection_layer = nn.Linear(
                    configs.d_model, 1, bias=True)
            else:
                self.projection_layer = nn.Linear(
                    configs.d_model, configs.c_out, bias=True)

                self.out_res_layers = torch.nn.ModuleList([
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** i),
                        configs.seq_len // (configs.down_sampling_window ** i),
                    )
                    for i in range(configs.down_sampling_layers + 1)
                ])

                self.regression_layers = torch.nn.ModuleList(
                    [
                        torch.nn.Linear(
                            configs.seq_len // (configs.down_sampling_window ** i),
                            configs.pred_len,
                        )
                        for i in range(configs.down_sampling_layers + 1)
                    ]
                )

        if self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
            if self.channel_independence:
                self.projection_layer = nn.Linear(
                    configs.d_model, 1, bias=True)
            else:
                self.projection_layer = nn.Linear(
                    configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                configs.d_model * configs.seq_len, configs.num_class)

    def out_projection(self, dec_out, i, out_res):
        dec_out = self.projection_layer(dec_out)
        out_res = out_res.permute(0, 2, 1)
        out_res = self.out_res_layers[i](out_res)
        out_res = self.regression_layers[i](out_res).permute(0, 2, 1)
        dec_out = dec_out + out_res
        return dec_out

    def pre_enc(self, x_list):
        if self.channel_independence:
            return (x_list, None)
        else:
            out1_list = []
            out2_list = []
            for x in x_list:
                x_1, x_2 = self.preprocess(x)
                out1_list.append(x_1)
                out2_list.append(x_2)
            return (out1_list, out2_list)

    def __multi_scale_process_inputs(self, x_enc, x_mark_enc):
        if self.configs.down_sampling_method == 'max':
            down_pool = torch.nn.MaxPool1d(self.configs.down_sampling_window, return_indices=False)
        elif self.configs.down_sampling_method == 'avg':
            down_pool = torch.nn.AvgPool1d(self.configs.down_sampling_window)
        elif self.configs.down_sampling_method == 'conv':
            padding = 1 if torch.__version__ >= '1.5.0' else 2
            down_pool = nn.Conv1d(in_channels=self.configs.enc_in, out_channels=self.configs.enc_in,
                                  kernel_size=3, padding=padding,
                                  stride=self.configs.down_sampling_window,
                                  padding_mode='circular',
                                  bias=False)
        else:
            return x_enc, x_mark_enc
        # B,T,C -> B,C,T
        x_enc = x_enc.permute(0, 2, 1)

        x_enc_ori = x_enc
        x_mark_enc_mark_ori = x_mark_enc

        x_enc_sampling_list = []
        x_mark_sampling_list = []
        x_enc_sampling_list.append(x_enc.permute(0, 2, 1))
        x_mark_sampling_list.append(x_mark_enc)

        for i in range(self.configs.down_sampling_layers):
            x_enc_sampling = down_pool(x_enc_ori)

            x_enc_sampling_list.append(x_enc_sampling.permute(0, 2, 1))
            x_enc_ori = x_enc_sampling

            if x_mark_enc is not None:
                x_mark_sampling_list.append(x_mark_enc_mark_ori[:, ::self.configs.down_sampling_window, :])
                x_mark_enc_mark_ori = x_mark_enc_mark_ori[:, ::self.configs.down_sampling_window, :]

        x_enc = x_enc_sampling_list
        x_mark_enc = x_mark_sampling_list if x_mark_enc is not None else None

        return x_enc, x_mark_enc

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        init_x_enc = x_enc

        x_enc, x_mark_enc = self.__multi_scale_process_inputs(x_enc, x_mark_enc)

        x_list = []
        x_mark_list = []
        if x_mark_enc is not None:
            for i, x, x_mark in zip(range(len(x_enc)), x_enc, x_mark_enc):
                B, T, N = x.size()
                x = self.normalize_layers[i](x, 'norm')
                if self.channel_independence:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                    x_list.append(x)
                    x_mark = x_mark.repeat(N, 1, 1)
                    x_mark_list.append(x_mark)
                else:
                    x_list.append(x)
                    x_mark_list.append(x_mark)
        else:
            for i, x in zip(range(len(x_enc)), x_enc, ):
                B, T, N = x.size()
                x = self.normalize_layers[i](x, 'norm')
                if self.channel_independence:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                x_list.append(x)

        # embedding
        enc_out_list = []

        x_list = self.pre_enc(x_list)

        if x_mark_enc is not None:
            for i, x, x_mark in zip(range(len(x_list[0])), x_list[0], x_mark_list):
                enc_out = self.enc_embedding(x, x_mark)  # [B,T,C]
                enc_out_list.append(enc_out)
        else:
            for i, x in zip(range(len(x_list[0])), x_list[0]):
                enc_out = self.enc_embedding(x, None)  # [B,T,C]
                enc_out_list.append(enc_out)

        # Past Decomposable Mixing as encoder for past
        for i in range(self.layer):
            enc_out_list = self.pdm_blocks[i](enc_out_list)


        # Future Multipredictor Mixing as decoder for future
        dec_out_list = self.future_multi_mixing(B, enc_out_list, x_list)
        # output the first point x and enc_out to csv by adding
        

        dec_out = torch.stack(dec_out_list, dim=-1).sum(-1)

        dec_out = self.normalize_layers[0](dec_out, 'denorm')
        
        # save trend
        if not self.training:
            output_trend_path = os.path.join('./test_results/',f'sl{self.configs.seq_len}_pl{self.configs.pred_len}_trends.csv')
            with open(output_trend_path, 'a') as f:
                writer = csv.writer(f)
                # x_list[0][:][dim] -> [B, T, 1]
                # x_list = [B/2, seq, 1]
                # enc_out_list[0][:][dim] -> [B, T, 1]
                # sum the last dimension of x_list[0][:] and enc_out_list[0][:]
                # x_list_output = x_list[0].sum(-1)
                init_x_enc_denorm = self.normalize_layers[0](init_x_enc, 'denorm')
                enc_out_list_denorm = self.normalize_layers[0](enc_out_list[0], 'denorm')
                for x_enc_denorm, enc_out_denorm in zip(init_x_enc_denorm, enc_out_list_denorm):
                    writer.writerow([x_enc_denorm[0][0].detach().cpu().item(), enc_out_denorm[0].sum(-1).detach().cpu().item()])
        
    
        return dec_out

    def future_multi_mixing(self, B, enc_out_list, x_list):
        dec_out_list = []
        if self.channel_independence:
            x_list = x_list[0]
            for i, enc_out in zip(range(len(x_list)), enc_out_list):
                dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(
                    0, 2, 1)  # align temporal dimension
                dec_out = self.projection_layer(dec_out)
                dec_out = dec_out.reshape(B, self.configs.c_out, self.pred_len).permute(0, 2, 1).contiguous()
                dec_out_list.append(dec_out)

        else:
            for i, enc_out, out_res in zip(range(len(x_list[0])), enc_out_list, x_list[1]):
                dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(
                    0, 2, 1)  # align temporal dimension
                dec_out = self.out_projection(dec_out, i, out_res)
                dec_out_list.append(dec_out)

        return dec_out_list

    def classification(self, x_enc, x_mark_enc):
        x_enc, _ = self.__multi_scale_process_inputs(x_enc, None)
        x_list = x_enc

        # embedding
        enc_out_list = []
        for x in x_list:
            enc_out = self.enc_embedding(x, None)  # [B,T,C]
            enc_out_list.append(enc_out)

        # MultiScale-CrissCrossAttention  as encoder for past
        for i in range(self.layer):
            enc_out_list = self.pdm_blocks[i](enc_out_list)

        enc_out = enc_out_list[0]
        # Output
        # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.act(enc_out)
        output = self.dropout(output)
        # zero-out padding embeddings
        output = output * x_mark_enc.unsqueeze(-1)
        # (batch_size, seq_length * d_model)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def anomaly_detection(self, x_enc):
        B, T, N = x_enc.size()
        x_enc, _ = self.__multi_scale_process_inputs(x_enc, None)

        x_list = []

        for i, x in zip(range(len(x_enc)), x_enc, ):
            B, T, N = x.size()
            x = self.normalize_layers[i](x, 'norm')
            if self.channel_independence:
                x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
            x_list.append(x)

        # embedding
        enc_out_list = []
        for x in x_list:
            enc_out = self.enc_embedding(x, None)  # [B,T,C]
            enc_out_list.append(enc_out)

        # MultiScale-CrissCrossAttention  as encoder for past
        for i in range(self.layer):
            enc_out_list = self.pdm_blocks[i](enc_out_list)

        dec_out = self.projection_layer(enc_out_list[0])
        dec_out = dec_out.reshape(B, self.configs.c_out, -1).permute(0, 2, 1).contiguous()

        dec_out = self.normalize_layers[0](dec_out, 'denorm')
        return dec_out

    def imputation(self, x_enc, x_mark_enc, mask):
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x_enc = x_enc - means
        x_enc = x_enc.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) /
                           torch.sum(mask == 1, dim=1) + 1e-5)
        stdev = stdev.unsqueeze(1).detach()
        x_enc /= stdev

        B, T, N = x_enc.size()
        x_enc, x_mark_enc = self.__multi_scale_process_inputs(x_enc, x_mark_enc)

        x_list = []
        x_mark_list = []
        if x_mark_enc is not None:
            for i, x, x_mark in zip(range(len(x_enc)), x_enc, x_mark_enc):
                B, T, N = x.size()
                if self.channel_independence:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                x_list.append(x)
                x_mark = x_mark.repeat(N, 1, 1)
                x_mark_list.append(x_mark)
        else:
            for i, x in zip(range(len(x_enc)), x_enc, ):
                B, T, N = x.size()
                if self.channel_independence:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                x_list.append(x)

        # embedding
        enc_out_list = []
        for x in x_list:
            enc_out = self.enc_embedding(x, None)  # [B,T,C]
            enc_out_list.append(enc_out)

        # MultiScale-CrissCrossAttention  as encoder for past
        for i in range(self.layer):
            enc_out_list = self.pdm_blocks[i](enc_out_list)

        dec_out = self.projection_layer(enc_out_list[0])
        dec_out = dec_out.reshape(B, self.configs.c_out, -1).permute(0, 2, 1).contiguous()

        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_mark_enc, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        else:
            raise ValueError('Other tasks implemented yet')
        

