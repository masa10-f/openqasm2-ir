OPENQASM 2.0;
include "qelib1.inc";
gate mygate(theta) a, b {
  rx(theta) a;
  cx a, b;
  ry(theta) b;
}
qreg q[2];
creg c[2];
mygate(pi/4) q[0], q[1];
measure q[0] -> c[0];
measure q[1] -> c[1];
