%% Set part kernel for head
kernel_size = 100;
head_height = 60;
head_width  = 40;
head_core_height = 0;
head_core_width  = 0;
h_gap = (kernel_size - head_height)/2;
w_gap = (kernel_size - head_width)/2;
h_core_gap = (head_height - head_core_height)/2;
w_core_gap = (head_width  - head_core_width)/2;

head_kernel = zeros(kernel_size);
head_kernel(h_gap+1 : h_gap+head_height, w_gap+1 : w_gap+head_width) = -1;
head_kernel(h_gap+h_core_gap+1 : h_gap+h_core_gap+head_core_height, ...
            w_gap+w_core_gap+1 : w_gap+w_core_gap+head_core_width) = 0;