// Implementation of a simple progressbar.
// @see https://stackoverflow.com/questions/14539867/
// ZHUO Xu @ https://github.com/z0gSh1u/nlm-cuda

#pragma once

#ifndef PROGRESSBAR_HPP
#define PROGRESSBAR_HPP

#include <iostream>
using std::cout;

class ProgressBar {
private:
  float _progress;
  int _barWidth;

public:
  ProgressBar(int barWidth = 40);
  void update(float delta);
  ~ProgressBar();
};

ProgressBar::ProgressBar(int barWidth) {
  this->_progress = 0;
  this->_barWidth = barWidth;
}

void ProgressBar::update(float delta) {
  this->_progress += delta;
  cout << "[";
  int pos = this->_barWidth * this->_progress;
  for (int i = 0; i < this->_barWidth; i++) {
    cout << (i < pos ? "=" : i == pos ? ">" : " ");
  }
  cout << "] " << int(this->_progress * 100.0) << " %\r";
  cout.flush();
}

ProgressBar::~ProgressBar() {}

#endif