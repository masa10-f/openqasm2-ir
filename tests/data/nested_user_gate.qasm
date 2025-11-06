OPENQASM 2.0;
include "qelib1.inc";
gate layer1(theta) a {
  rx(theta) a;
  ry(theta) a;
}
gate layer2(phi) a, b {
  layer1(phi) a;
  cx a, b;
  layer1(phi) b;
}
qreg q[2];
creg c[2];
layer2(pi/3) q[0], q[1];
measure q -> c;
