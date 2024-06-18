//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include <iostream>
#include <cstdlib>
#include <vector>
#include <assert.h>
#include <chrono>
#include <algorithm>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "memory.h"
#include "bmruntime_interface.h"
#include <getopt.h>
#include <stdio.h>
#include <inttypes.h>
#include <random>
#include <numeric>

namespace py = pybind11;

static const uint16_t ATTENTION_MASK = 0xF0E2;

class TTSLlama
{
public:
  void init(const std::vector<int> &devid, std::string model_path);
  void deinit();

  int forward_first_text(std::vector<int> &tokens);
  int forward_next_text();
  std::vector<int> generate_text(std::vector<int> &history_tokens, int EOS, float tempreature);

  py::dict forward_first_code(std::vector<int> &tokens, int idx_spk, std::vector<uint16_t> spk_emb);
  py::dict forward_next_code();
  py::dict generate_code(std::vector<int> &history_tokens, int spk_idx, std::vector<uint16_t> &spk_emb, int EOS, float tempreture);

  std::mt19937 sgen;
  TTSLlama() : sgen(std::random_device()()){};

private:
  void net_launch(const bm_net_info_t *net, int stage_idx = 0);
  inline void d2d(bm_device_mem_t &dst, bm_device_mem_t &src);

  void head_launch(const bm_net_info_t *net, bm_device_mem_t &logits_mem);
  int greedy_search(const bm_net_info_t *net, bm_device_mem_t &logits_mem);
  int penalty_sample(const bm_net_info_t *net, bm_device_mem_t &logits_mem);

public:
  int token_length;
  int hidden_size; // read from bmodel
  int SEQLEN;      // read from bmodel
  int NUM_LAYERS;  // read from bmodel
  bool io_alone;
  std::vector<int> visited_tokens;

  // generation
  float temperature;
  float top_p;
  float repeat_penalty;
  int repeat_last_n;
  int max_new_tokens;

private:
  std::vector<bm_handle_t> handles;
  bm_handle_t bm_handle;
  void *p_bmrt;
  std::vector<const bm_net_info_t *> net_blocks;
  std::vector<const bm_net_info_t *> net_blocks_cache;
  const bm_net_info_t *net_embed_text, *net_embed_code;
  const bm_net_info_t *net_embed_text_cache, *net_embed_code_cache;
  const bm_net_info_t *net_lm_text, *net_lm_code, *net_penalty_sample_head_text, *net_penalty_sample_head_code;
  std::vector<bm_device_mem_t> past_key;
  std::vector<bm_device_mem_t> past_value;
};

void TTSLlama::net_launch(const bm_net_info_t *net, int stage_idx)
{
  std::vector<bm_tensor_t> in_tensors(net->input_num);
  std::vector<bm_tensor_t> out_tensors(net->output_num);

  for (int i = 0; i < net->input_num; i++)
  {
    bmrt_tensor_with_device( // 将已经分配的设备内存（bm_device_mem_t）与一个张量结构体关联起来，并设置张量的数据类型和形状。
        &in_tensors[i], net->stages[stage_idx].input_mems[i],
        net->input_dtypes[i], net->stages[stage_idx].input_shapes[i]);
  }
  for (int i = 0; i < net->output_num; i++)
  {
    bmrt_tensor_with_device(
        &out_tensors[i], net->stages[stage_idx].output_mems[i],
        net->output_dtypes[i], net->stages[stage_idx].output_shapes[i]);
  }
  auto ret = bmrt_launch_tensor_ex(p_bmrt, net->name, in_tensors.data(),
                                   net->input_num, out_tensors.data(),
                                   net->output_num, true, false);
  assert(ret);
  bm_thread_sync(bm_handle);
}

void TTSLlama::d2d(bm_device_mem_t &dst, bm_device_mem_t &src)
{
  bm_memcpy_d2d_byte(bm_handle, dst, 0, src, 0, bm_mem_get_device_size(src));
}

void TTSLlama::init(const std::vector<int> &devices, std::string model_path)
{

  // request bm_handle
  std::cout << "Device [ ";
  for (auto d : devices)
  {
    std::cout << d << " ";
  }
  std::cout << "] loading ....\n";
  for (auto d : devices)
  {
    bm_handle_t h;
    bm_status_t status = bm_dev_request(&h, d);
    assert(BM_SUCCESS == status);
    handles.push_back(h);
  }
  bm_handle = handles[0];

  // create bmruntime
#ifdef SOC_TARGET
  p_bmrt = bmrt_create(handles[0]);
#else
  p_bmrt = bmrt_create_ex(handles.data(), handles.size());
#endif
  assert(NULL != p_bmrt);

  // load bmodel by file
  printf("Model[%s] loading ....\n", model_path.c_str());
  bool ret = bmrt_load_bmodel(p_bmrt, model_path.c_str());
  assert(true == ret);
  printf("Done!\n");

  // net embed and lm_head
  net_embed_text = bmrt_get_network_info(p_bmrt, "embedding_text");
  net_embed_text_cache = bmrt_get_network_info(p_bmrt, "embedding_text_cache");
  net_embed_code = bmrt_get_network_info(p_bmrt, "embedding_code");
  net_embed_code_cache = bmrt_get_network_info(p_bmrt, "embedding_code_cache");

  net_lm_text = bmrt_get_network_info(p_bmrt, "lm_head_text");
  net_lm_code = bmrt_get_network_info(p_bmrt, "lm_head_code");

  net_penalty_sample_head_text = bmrt_get_network_info(p_bmrt, "penalty_sample_head_text");
  net_penalty_sample_head_code = bmrt_get_network_info(p_bmrt, "penalty_sample_head_code");

  SEQLEN = net_embed_text->stages[0].input_shapes[0].dims[1]; // real seqlen
  printf("SEQLEN: %d\n", SEQLEN);
  auto num_nets = bmrt_get_network_number(p_bmrt);

  NUM_LAYERS = (num_nets - 8) / 2;
  hidden_size = net_embed_text->stages[0].output_shapes[0].dims[2];
  printf("NUM_LAYERS: %d, hidden_size: %d\n", NUM_LAYERS, hidden_size);

  // resize
  visited_tokens.resize(SEQLEN);

  // net blocks
  for (int i = 0; i < NUM_LAYERS; i++)
  {
    auto block_name = "block_" + std::to_string(i);
    auto cache_name = "block_cache_" + std::to_string(i);
    net_blocks.emplace_back(bmrt_get_network_info(p_bmrt, block_name.c_str()));
    net_blocks_cache.emplace_back(
        bmrt_get_network_info(p_bmrt, cache_name.c_str()));
  }

  // kv cache
  past_key.resize(NUM_LAYERS);
  past_value.resize(NUM_LAYERS);
  auto addr_mode = net_blocks_cache[0]->addr_mode;
  io_alone = addr_mode == 1;
  for (int i = 0; i < NUM_LAYERS; i++)
  {
    assert(addr_mode == net_blocks_cache[i]->addr_mode);
    if (io_alone)
    {
      past_key[i] = net_blocks_cache[i]->stages[0].input_mems[3];
      past_value[i] = net_blocks_cache[i]->stages[0].input_mems[4];
    }
    else
    {
      auto ret = bm_malloc_device_byte(bm_handle, &past_key[i],
                                       net_blocks_cache[i]->max_input_bytes[3]);
      assert(BM_SUCCESS == ret);
      ret = bm_malloc_device_byte(bm_handle, &past_value[i],
                                  net_blocks_cache[i]->max_input_bytes[4]);
      assert(BM_SUCCESS == ret);
    }
  }
}

void TTSLlama::deinit()
{
  if (false == io_alone)
  {
    for (int i = 0; i < NUM_LAYERS; i++)
    {
      bm_free_device(bm_handle, past_key[i]);
      bm_free_device(bm_handle, past_value[i]);
    }
  }
  bmrt_destroy(p_bmrt);
  for (auto h : handles)
  {
    bm_dev_free(h);
  }
}

void TTSLlama::head_launch(const bm_net_info_t *net, bm_device_mem_t &logits_mem)
{
  std::vector<bm_tensor_t> in_tensors(net->input_num);
  std::vector<bm_tensor_t> out_tensors(net->output_num);

  bmrt_tensor_with_device( // #############
      &in_tensors[0], logits_mem, // size = 84712 21178*4 for text generation; 
      net->input_dtypes[0], net->stages[0].input_shapes[0]); //

  for (int i = 1; i < net->input_num; i++)
  {
    bmrt_tensor_with_device(
        &in_tensors[i], net->stages[0].input_mems[i],
        net->input_dtypes[i], net->stages[0].input_shapes[i]);
  }
  for (int i = 0; i < net->output_num; i++)
  {
    bmrt_tensor_with_device(
        &out_tensors[i], net->stages[0].output_mems[i],
        net->output_dtypes[i], net->stages[0].output_shapes[i]);
  }
  auto ret = bmrt_launch_tensor_ex(p_bmrt, net->name, in_tensors.data(),
                                   net->input_num, out_tensors.data(),
                                   net->output_num, true, false);
  assert(ret);
  bm_thread_sync(bm_handle);
}

// int TTSLlama::greedy_search(const bm_net_info_t *net, bm_device_mem_t &logits_mem) {
//   auto &out_mem = net->stages[0].output_mems[0];
//   head_launch(net, logits_mem);
//   int token = 0;
//   bm_memcpy_d2s(bm_handle, (void *)&token, out_mem); // device mem to system mem
//   return token;
// }

int TTSLlama::penalty_sample(const bm_net_info_t *net, bm_device_mem_t &logits_mem)
{
  auto &in1_mem = net->stages[0].input_mems[1];
  auto &in2_mem = net->stages[0].input_mems[2];
  auto &in3_mem = net->stages[0].input_mems[3];
  auto &in4_mem = net->stages[0].input_mems[4];
  auto &out0_mem = net->stages[0].output_mems[0];
  auto &out1_mem = net->stages[0].output_mems[1];

  // repeat_penalty + top_p + top_k + temperature
  std::vector<int> generated_tokens(SEQLEN, visited_tokens[token_length - 1]);
  repeat_last_n = std::min(repeat_last_n, token_length);
  std::copy(visited_tokens.begin() + token_length - repeat_last_n,
            visited_tokens.begin() + token_length,
            generated_tokens.begin());
  bm_memcpy_s2d(bm_handle, in1_mem, (void *)generated_tokens.data());
  bm_memcpy_s2d(bm_handle, in2_mem, (void *)&top_p);
  bm_memcpy_s2d(bm_handle, in3_mem, (void *)&temperature);
  bm_memcpy_s2d(bm_handle, in4_mem, (void *)&repeat_penalty);

  // inference
  head_launch(net, logits_mem);

  // get logit & token
  int candidate_num = net->stages[0].output_shapes[0].dims[1];
  std::vector<float> probs(candidate_num);
  bm_memcpy_d2s(bm_handle, probs.data(), out0_mem);
  std::vector<int> tokens(candidate_num);
  bm_memcpy_d2s(bm_handle, tokens.data(), out1_mem);

  // penalty_sample
  std::discrete_distribution<> dist(probs.begin(), probs.end());
  return tokens[dist(sgen)];
}

int TTSLlama::forward_first_text(std::vector<int> &tokens)
{
  std::vector<int> position_id(SEQLEN, 0);
  std::vector<uint16_t> attention_mask(SEQLEN * SEQLEN, ATTENTION_MASK);
  std::copy(tokens.begin(), tokens.end(), visited_tokens.data());

  token_length = tokens.size();

  for (int i = 0; i < token_length; i++)
  {
    position_id[i] = i;
  }
  for (int i = 0; i < token_length; i++)
  {
    for (int j = 0; j < SEQLEN; j++)
    {
      if (j <= i)
      {
        attention_mask[i * SEQLEN + j] = 0;
      }
    }
  }

  // forward embeding
  auto &in_mem = net_embed_text->stages[0].input_mems[0];
  auto &out_mem = net_embed_text->stages[0].output_mems[0];
  bm_memcpy_s2d(bm_handle, in_mem, (void *)visited_tokens.data());
  net_launch(net_embed_text); // prefil embedding

  // forward blocks
  for (int idx = 0; idx < NUM_LAYERS; idx++)
  {
    auto &in0_mem = net_blocks[idx]->stages[0].input_mems[0];
    auto &in1_mem = net_blocks[idx]->stages[0].input_mems[1];
    auto &in2_mem = net_blocks[idx]->stages[0].input_mems[2];
    d2d(in0_mem, out_mem);
    if (idx == 0)
    {
      // only first time need copy
      bm_memcpy_s2d(bm_handle, in1_mem, (void *)position_id.data());
      bm_memcpy_s2d(bm_handle, in2_mem, (void *)attention_mask.data());
    }
    net_launch(net_blocks[idx]);
    out_mem = net_blocks[idx]->stages[0].output_mems[0];
    d2d(past_key[idx], net_blocks[idx]->stages[0].output_mems[1]);
    d2d(past_value[idx], net_blocks[idx]->stages[0].output_mems[2]);
  }

  // forward lmhead
  int bytes = out_mem.size / SEQLEN;
  auto &lm_in_mem = net_lm_text->stages[0].input_mems[0];
  auto &lm_out_mem = net_lm_text->stages[0].output_mems[0];
  bm_memcpy_d2d_byte(bm_handle, lm_in_mem, 0, out_mem,
                     (token_length - 1) * bytes, bytes);
  net_launch(net_lm_text);

  int token = 0;
  token = penalty_sample(net_penalty_sample_head_text, lm_out_mem); //size = 84712 

  visited_tokens[token_length] = token;
  token_length += 1;
  return token;
}

int TTSLlama::forward_next_text()
{
  int cur_token = visited_tokens[token_length - 1];

  std::vector<uint16_t> attention_mask(SEQLEN + 1, 0);
  for (int i = token_length - 1; i < SEQLEN; i++)
  {
    attention_mask[i] = ATTENTION_MASK;
  }
  int32_t position_id = token_length - 1;

  // embedding
  auto &in_mem = net_embed_text_cache->stages[0].input_mems[0];
  auto &out_mem = net_embed_text_cache->stages[0].output_mems[0];
  bm_memcpy_s2d(bm_handle, in_mem, (void *)&cur_token);
  net_launch(net_embed_text_cache);

  // blocks
  int bytes =
      bm_mem_get_device_size(net_blocks_cache[0]->stages[0].output_mems[1]);
  int token_offset = (token_length - 1) * bytes;
  for (int idx = 0; idx < NUM_LAYERS; idx++)
  {
    auto &in0_mem = net_blocks_cache[idx]->stages[0].input_mems[0];
    auto &in1_mem = net_blocks_cache[idx]->stages[0].input_mems[1];
    auto &in2_mem = net_blocks_cache[idx]->stages[0].input_mems[2];
    auto &in3_mem = net_blocks_cache[idx]->stages[0].input_mems[3];
    auto &in4_mem = net_blocks_cache[idx]->stages[0].input_mems[4];
    auto &out0_mem = net_blocks_cache[idx]->stages[0].output_mems[0];
    auto &out1_mem = net_blocks_cache[idx]->stages[0].output_mems[1];
    auto &out2_mem = net_blocks_cache[idx]->stages[0].output_mems[2];
    d2d(in0_mem, out_mem);
    if (io_alone)
    {
      if (idx == 0)
      {
        bm_memcpy_s2d(bm_handle, in1_mem, (void *)&position_id);
        bm_memcpy_s2d(bm_handle, in2_mem, (void *)attention_mask.data());
      }
      else
      {
        d2d(in1_mem, net_blocks_cache[0]->stages[0].input_mems[1]);
        d2d(in2_mem, net_blocks_cache[0]->stages[0].input_mems[2]);
      }
    }
    else
    {
      if (idx == 0)
      {
        bm_memcpy_s2d(bm_handle, in1_mem, (void *)&position_id);
        bm_memcpy_s2d(bm_handle, in2_mem, (void *)attention_mask.data());
      }
      d2d(in3_mem, past_key[idx]);
      d2d(in4_mem, past_value[idx]);
    }
    net_launch(net_blocks_cache[idx]);
    out_mem = out0_mem;
    bm_memcpy_d2d_byte(bm_handle, past_key[idx], token_offset, out1_mem, 0,
                       bytes);
    bm_memcpy_d2d_byte(bm_handle, past_value[idx], token_offset, out2_mem, 0,
                       bytes);
  }

  // forward lmhead
  auto &lm_in_mem = net_lm_text->stages[0].input_mems[0];
  auto &lm_out_mem = net_lm_text->stages[0].output_mems[0];
  d2d(lm_in_mem, out_mem);
  net_launch(net_lm_text);

  int token = 0;
  token = penalty_sample(net_penalty_sample_head_text, lm_out_mem);

  visited_tokens[token_length] = token;
  token_length += 1;
  return token;
}

std::vector<int> TTSLlama::generate_text(std::vector<int> &history_tokens, int EOS, float temp_tempreature)
{
  if (history_tokens.empty())
  {
    printf("Sorry: your text is empty!!\n");
    history_tokens.clear();
    return {};
  }

  // make sure token not too large
  if ((int)history_tokens.size() > SEQLEN - 10)
  {
    history_tokens.clear();
    printf("Error: your original text is too large!\n");
    return {};
  }
  std::fill(visited_tokens.begin(), visited_tokens.end(), 0);
  temperature = temp_tempreature;

  std::vector<int> result_tokens;
  int token = forward_first_text(history_tokens);
  while (token != EOS && token_length < SEQLEN)
  {
    result_tokens.emplace_back(token);
    token = forward_next_text();
  }
  return result_tokens;
}

py::dict TTSLlama::forward_first_code(std::vector<int> &tokens, int idx_spk, std::vector<uint16_t> spk_emb)
{
  printf("forward_first_code\n");
  std::vector<int> position_id(SEQLEN, 0);
  std::vector<uint16_t> attention_mask(SEQLEN * SEQLEN, ATTENTION_MASK);
  std::copy(tokens.begin(), tokens.end(), visited_tokens.data());

  token_length = tokens.size();

  for (int i = 0; i < token_length; i++)
  {
    position_id[i] = i;
  }
  for (int i = 0; i < token_length; i++)
  {
    for (int j = 0; j < SEQLEN; j++)
    {
      if (j <= i)
      {
        attention_mask[i * SEQLEN + j] = 0;
      }
    }
  }

  // forward embeding
  auto &in_mem = net_embed_code->stages[0].input_mems[0];
  auto &out_mem = net_embed_code->stages[0].output_mems[0];
  bm_memcpy_s2d(bm_handle, in_mem, (void *)visited_tokens.data());
  printf("token_length: %d\n", token_length);
  net_launch(net_embed_code); // prefil embedding

  // the embedding of empty_spk --> spk_emb
  if (idx_spk != -1)
  {
    int byte_size = 2;
    // switch(out_mem->dtype) {
    //     case bmruntime::BM_FLOAT16:
    //         byte_size = 2;
    //         break;
    //     default:
    //         std::cout << "Data type is unknown" << std::endl;
    //         exit(-1);
    // }
    bm_memcpy_s2d_partial_offset(bm_handle, out_mem, (void *)spk_emb.data(), hidden_size * byte_size, idx_spk * hidden_size * byte_size); // bytesize, offset
  }

  // forward blocks
  for (int idx = 0; idx < NUM_LAYERS; idx++)
  {
    auto &in0_mem = net_blocks[idx]->stages[0].input_mems[0];
    auto &in1_mem = net_blocks[idx]->stages[0].input_mems[1];
    auto &in2_mem = net_blocks[idx]->stages[0].input_mems[2];
    d2d(in0_mem, out_mem);
    if (idx == 0)
    {
      // only first time need copy
      bm_memcpy_s2d(bm_handle, in1_mem, (void *)position_id.data());
      bm_memcpy_s2d(bm_handle, in2_mem, (void *)attention_mask.data());
    }
    net_launch(net_blocks[idx]);
    out_mem = net_blocks[idx]->stages[0].output_mems[0]; // hiddens
    d2d(past_key[idx], net_blocks[idx]->stages[0].output_mems[1]);
    d2d(past_value[idx], net_blocks[idx]->stages[0].output_mems[2]);
  }

  std::vector<uint16_t> last_hidden(hidden_size, 0);
  bm_memcpy_d2s(bm_handle, (void *)last_hidden.data(), net_blocks[NUM_LAYERS - 1]->stages[0].output_mems[0]);

  // forward lmhead
  int bytes = out_mem.size / SEQLEN;
  auto &lm_in_mem = net_lm_code->stages[0].input_mems[0];
  auto &lm_out_mem = net_lm_code->stages[0].output_mems[0];
  bm_memcpy_d2d_byte(bm_handle, lm_in_mem, 0, out_mem,
                     (token_length - 1) * bytes, bytes);
  net_launch(net_lm_code);

  int token = 0;
  token = penalty_sample(net_penalty_sample_head_code, lm_out_mem);

  visited_tokens[token_length] = token;
  token_length += 1;
  py::dict result;
  result["token"] = token;
  result["last_hidden"] = last_hidden;
  return result;
}

py::dict TTSLlama::forward_next_code()
{
  int cur_token = visited_tokens[token_length - 1];

  std::vector<uint16_t> attention_mask(SEQLEN + 1, 0);
  for (int i = token_length - 1; i < SEQLEN; i++)
  {
    attention_mask[i] = ATTENTION_MASK;
  }
  int32_t position_id = token_length - 1;

  // embedding
  auto &in_mem = net_embed_code_cache->stages[0].input_mems[0];
  auto &out_mem = net_embed_code_cache->stages[0].output_mems[0];
  bm_memcpy_s2d(bm_handle, in_mem, (void *)&cur_token);
  net_launch(net_embed_code_cache);

  // blocks
  int bytes =
      bm_mem_get_device_size(net_blocks_cache[0]->stages[0].output_mems[1]);
  int token_offset = (token_length - 1) * bytes;
  for (int idx = 0; idx < NUM_LAYERS; idx++)
  {
    auto &in0_mem = net_blocks_cache[idx]->stages[0].input_mems[0];
    auto &in1_mem = net_blocks_cache[idx]->stages[0].input_mems[1];
    auto &in2_mem = net_blocks_cache[idx]->stages[0].input_mems[2];
    auto &in3_mem = net_blocks_cache[idx]->stages[0].input_mems[3];
    auto &in4_mem = net_blocks_cache[idx]->stages[0].input_mems[4];
    auto &out0_mem = net_blocks_cache[idx]->stages[0].output_mems[0];
    auto &out1_mem = net_blocks_cache[idx]->stages[0].output_mems[1];
    auto &out2_mem = net_blocks_cache[idx]->stages[0].output_mems[2];
    d2d(in0_mem, out_mem);
    if (io_alone)
    {
      if (idx == 0)
      {
        bm_memcpy_s2d(bm_handle, in1_mem, (void *)&position_id);
        bm_memcpy_s2d(bm_handle, in2_mem, (void *)attention_mask.data());
      }
      else
      {
        d2d(in1_mem, net_blocks_cache[0]->stages[0].input_mems[1]);
        d2d(in2_mem, net_blocks_cache[0]->stages[0].input_mems[2]);
      }
    }
    else
    {
      if (idx == 0)
      {
        bm_memcpy_s2d(bm_handle, in1_mem, (void *)&position_id);
        bm_memcpy_s2d(bm_handle, in2_mem, (void *)attention_mask.data());
      }
      d2d(in3_mem, past_key[idx]);
      d2d(in4_mem, past_value[idx]);
    }
    net_launch(net_blocks_cache[idx]);
    out_mem = out0_mem;
    bm_memcpy_d2d_byte(bm_handle, past_key[idx], token_offset, out1_mem, 0,
                       bytes);
    bm_memcpy_d2d_byte(bm_handle, past_value[idx], token_offset, out2_mem, 0,
                       bytes);
  }

  std::vector<uint16_t> last_hidden(hidden_size, 0);
  bm_memcpy_d2s(bm_handle, (void *)last_hidden.data(), out_mem);

  // forward lmhead
  auto &lm_in_mem = net_lm_text->stages[0].input_mems[0];
  auto &lm_out_mem = net_lm_text->stages[0].output_mems[0];
  d2d(lm_in_mem, out_mem);
  net_launch(net_lm_text);

  int token = 0;
  token = penalty_sample(net_penalty_sample_head_code, lm_out_mem);

  visited_tokens[token_length] = token;
  token_length += 1;
  py::dict result;
  result["token"] = token;
  result["last_hidden"] = last_hidden;
  return result;
}

py::dict TTSLlama::generate_code(std::vector<int> &history_tokens, int idx_spk, std::vector<uint16_t> &spk_emb, int EOS, float temp_tempreature)
{
  printf("generate code\n");
  py::dict result;
  if (history_tokens.empty())
  {
    printf("Sorry: your text is empty!!\n");
    history_tokens.clear();
    return result;
  }

  // make sure token not too largace
  if ((int)history_tokens.size() > SEQLEN - 10)
  {
    history_tokens.clear();
    printf("Error: your text is too large!\n");
    return result;
  }

  temperature = temp_tempreature;
  printf("temperature: %f\n", temperature);
  std::vector<int> result_tokens;
  std::vector<std::vector<uint16_t>> result_hiddens;
  py::dict curr_res = forward_first_code(history_tokens, idx_spk, spk_emb);
  int token = curr_res["token"].cast<int>();
  while (token != EOS && token_length < SEQLEN)
  {
    result_tokens.emplace_back(token);
    result_hiddens.emplace_back(curr_res["hiddens"].cast<std::vector<uint16_t>>());
    curr_res = forward_next_code();
    token = curr_res["token"].cast<int>();
  }
  result["tokens"] = result_tokens;
  result["hiddens"] = result_hiddens;
  return result;
}

PYBIND11_MODULE(llama, m)
{
  pybind11::class_<TTSLlama>(m, "TTSLlama")
      .def(pybind11::init<>())
      .def("init", &TTSLlama::init)
      .def("deinit", &TTSLlama::deinit)
      .def("forward_first_text", &TTSLlama::forward_first_text)
      .def("forward_next_text", &TTSLlama::forward_next_text)
      .def("generate_text", &TTSLlama::generate_text)
      .def("forward_first_code", &TTSLlama::forward_first_code)
      .def("forward_next_code", &TTSLlama::forward_next_code)
      .def("generate_code", &TTSLlama::generate_code)
      .def_readwrite("SEQLEN", &TTSLlama::SEQLEN) // read SEQLEN in pipeline.py
      .def_readwrite("token_length", &TTSLlama::token_length)
      .def_readwrite("temperature", &TTSLlama::temperature)
      .def_readwrite("top_p", &TTSLlama::top_p)
      .def_readwrite("repeat_penalty", &TTSLlama::repeat_penalty)
      .def_readwrite("repeat_last_n", &TTSLlama::repeat_last_n)
      .def_readwrite("max_new_tokens", &TTSLlama::max_new_tokens);
}