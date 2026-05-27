function AArun_batch_metrics(root_folder, fused_root, output_folder)

grey_level = 256;

% ── Discover source image folders ─────────────────────────────────────────────
src_entries = dir(root_folder);
src_entries = src_entries([src_entries.isdir] & ~startsWith({src_entries.name}, '.'));

assert(numel(src_entries) == 2, ...
    'Expected exactly 2 source folders in "%s", found %d.', root_folder, numel(src_entries));

src_dirs = { fullfile(src_entries(1).folder, src_entries(1).name), ...
             fullfile(src_entries(2).folder, src_entries(2).name) };

% ── Discover method subfolders inside fused root ───────────────────────────────
fused_entries = dir(fused_root);
fused_entries = fused_entries([fused_entries.isdir] & ~startsWith({fused_entries.name}, '.'));

assert(~isempty(fused_entries), 'No method folders found inside "%s".', fused_root);

fprintf('Source 1   : %s\n', src_dirs{1});
fprintf('Source 2   : %s\n', src_dirs{2});
fprintf('Fused root : %s\n', fused_root);
for f = 1:numel(fused_entries)
    fprintf('  Method %d : %s\n', f, fused_entries(f).name);
end

% ── Output directory ──────────────────────────────────────────────────────────
if ~exist(output_folder, 'dir'), mkdir(output_folder); end

% ── Supported image extensions ────────────────────────────────────────────────
IMG_EXT = {'*.png','*.jpg','*.jpeg','*.bmp','*.tif','*.tiff'};

% ── Column definitions ────────────────────────────────────────────────────────
% INFO | IMAGE | STRUCTURE | VISUAL | QUALITY
all_cols = { ...
    'ImageName', ...
    'EN',  'MI',   'FMI',  'SCD',      ... % INFO
    'Qabf','SF',                        ... % IMAGE
    'SSIM','MS_SSIM',                   ... % STRUCTURE
    'VIF', 'VIFF',                      ... % VISUAL
    'SD',  'MLI',  'AG',   'PSNR'      ... % QUALITY
};

summary_cols = { ...
    'Model', ...
    'EN',  'MI',   'FMI',  'SCD',      ...
    'Qabf','SF',                        ...
    'SSIM','MS_SSIM',                   ...
    'VIF', 'VIFF',                      ...
    'SD',  'MLI',  'AG',   'PSNR'      ...
};

% ── Summary accumulator (one row per method) ──────────────────────────────────
summary_data = {};

% ── Process each method folder ────────────────────────────────────────────────
for fi = 1:numel(fused_entries)
    method_name = fused_entries(fi).name;
    fused_dir   = fullfile(fused_root, method_name);

    fprintf('\n==============================\n');
    fprintf('Processing: %s\n', method_name);
    fprintf('==============================\n');

    % Collect image files
    img_files = [];
    for e = 1:numel(IMG_EXT)
        found = dir(fullfile(fused_dir, IMG_EXT{e}));
        img_files = [img_files; found]; %#ok<AGROW>
    end

    if isempty(img_files)
        warning('No images found in %s, skipping.', fused_dir);
        continue;
    end

    result_data = {};

    for ii = 1:numel(img_files)
        img_name = img_files(ii).name;
        fprintf('[%d/%d] %s ... ', ii, numel(img_files), img_name);

        % ── Load images ───────────────────────────────────────────────────
        path1 = fullfile(src_dirs{1}, img_name);
        path2 = fullfile(src_dirs{2}, img_name);
        pathF = fullfile(fused_dir,   img_name);

        if ~isfile(path1) || ~isfile(path2)
            fprintf('SKIPPED (source image not found)\n');
            continue;
        end

        img1  = ensureGray(imread(path1));
        img2  = ensureGray(imread(path2));
        img_f = ensureGray(imread(pathF));

        [s1, s2] = size(img1);

        img1_int  = img1;
        img2_int  = img2;
        img_f_int = img_f;

        img1_float  = im2double(img1) * 255.0;
        img2_float  = im2double(img2) * 255.0;
        img_f_float = im2double(img_f) * 255.0;

        imgSeq = zeros(s1, s2, 2);
        imgSeq(:,:,1) = img1_float;
        imgSeq(:,:,2) = img2_float;

        % try
            % ── INFO ──────────────────────────────────────────────────────
            EN   = EN_metrics(img_f_int);
            MI   = MI_metrics(img1_int, img2_int, img_f_int, grey_level);
            FMI  = FMI_metrics(img1_float, img2_float, img_f_float);         % pixel-domain
            SCD  = SCD_metrics(img1_float, img2_float, img_f_float);

            % ── IMAGE ─────────────────────────────────────────────────────
            [Qabf, ~, ~, ~] = Petrovic_metrics(img_f_float, img1_float, img2_float);
            SF   = SF_metrics(img_f_float);

            % ── STRUCTURE ─────────────────────────────────────────────────
            SSIM    = SSIM_metrics(img1_float, img2_float, img_f_float);
            imgSeq = cat(3, img1_float, img2_float);
            [MS_SSIM, ~, ~] = MEF_SSIM_metrics(imgSeq, img_f_float);

            % ── VISUAL ────────────────────────────────────────────────────
            VIF  = VIF_metrics(img1_float, img_f_float) + VIF_metrics(img2_float, img_f_float);
            VIFF = VIFF_metrics(img1_float, img2_float, img_f_float);

            % ── QUALITY ───────────────────────────────────────────────────
            SD   = SD_metrics(img_f_float);
            MLI  = MLI_metrics(img1_float, img2_float, img_f_float);
            AG   = AG_metrics(img_f_float);
            PSNR = PSNR_metrics(img1_float, img2_float, img_f_float);

            result_data(end+1,:) = { ...
                img_name, ...
                EN, MI, FMI, SCD, ...
                Qabf, SF, ...
                SSIM, MS_SSIM, ...
                VIF, VIFF, ...
                SD, MLI, AG, PSNR ...
            }; %#ok<AGROW>

            fprintf('done\n');

        % catch err
        %     warning('\nMetrics failed for %s: %s', img_name, err.message);
        % end

    end % image loop

    % ── Save per-method CSV ───────────────────────────────────────────────────
    if ~isempty(result_data)
        T     = cell2table(result_data, 'VariableNames', all_cols);
        T_avg = computeAverageRow(T, all_cols, 'AVERAGE');

        out_csv = fullfile(output_folder, sprintf('%s_metrics.csv', method_name));
        writetable([T; T_avg], out_csv);
        fprintf('Results saved to: %s\n', out_csv);

        % Accumulate summary row
        sum_row            = T_avg;
        sum_row.ImageName  = {method_name};
        summary_data(end+1,:) = table2cell(sum_row); %#ok<AGROW>
    end

end % method loop

% ── Write summary CSV ────────────────────────────────────────────────────────
if numel(fused_entries) > 1 && ~isempty(summary_data)
    T_summary = cell2table(summary_data, 'VariableNames', summary_cols);
    summary_csv = fullfile(output_folder, 'summary_metrics.csv');
    writetable(T_summary, summary_csv);
    fprintf('\nSummary saved to: %s\n', summary_csv);
end

fprintf('\nAll done. Results in: %s\n', output_folder);

end % run_evaluation


% ═════════════════════════════════════════════════════════════════════════════
%  Helper functions
% ═════════════════════════════════════════════════════════════════════════════

function img = ensureGray(img)
    if size(img, 3) > 2
        img = rgb2gray(img);
    end
end

function avg_row = computeAverageRow(T, col_names, label)
    avg_cell    = cell(1, numel(col_names));
    avg_cell{1} = label;
    for c = 2:numel(col_names)
        col_data = T.(col_names{c});
        if isnumeric(col_data)
            avg_cell{c} = mean(col_data, 'omitnan');
        else
            avg_cell{c} = NaN;
        end
    end
    avg_row = cell2table(avg_cell, 'VariableNames', col_names);
end