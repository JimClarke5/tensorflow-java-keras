/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.tensorflow.keras.utils;

import java.io.PrintWriter;
import java.util.concurrent.atomic.AtomicInteger;
import static org.junit.jupiter.api.Assertions.assertEquals;
import org.tensorflow.DataType;
import org.tensorflow.EagerSession;
import org.tensorflow.Output;
import org.tensorflow.Session;
import org.tensorflow.op.Ops;
import org.tensorflow.types.TFloat32;
import org.tensorflow.types.TFloat64;
import org.tensorflow.types.TInt32;
import org.tensorflow.types.TInt64;
import org.tensorflow.types.family.TNumber;

/**
 *
 * @author Jim Clarke
 */
public class EagerTestSession extends TestSession {

    private final EagerSession session;
    private final Ops tf;

    public EagerTestSession() {
        this.session = EagerSession.create();
        this.tf = Ops.create(session).withName("test");
    }

    public Ops getTF() {
        return tf;
    }

    public EagerSession getSession() {
        return session;
    }

    @Override
    public void close() {
        session.close();
    }

    @Override
    public boolean isEager() {
        return true;
    }

    @Override
    public Session getGraphSession() {
        return null;
    }

    @Override
    public EagerSession getEagerSession() {
        return this.session;
    }

    @Override
    public <T extends TNumber> void evaluate(Number[] expected, Output<T> input) {
        boolean scalarExpected = expected.length == 1;
        DataType dtype = input.dataType();
        if (dtype == TFloat32.DTYPE) {
            Output<TFloat32> o = (Output<TFloat32>) input;
            AtomicInteger index = new AtomicInteger();
            if (debug) {
                o.data().scalars().forEach(f -> {
                    System.out.printf("%d). %f\n", index.incrementAndGet(), f.getFloat());
                });
            }
            index.set(0);
            o.data().scalars().forEach(f -> {
                assertEquals(expected[index.get()].floatValue(), f.getFloat(), epsilon);
                if (!scalarExpected) {
                    index.incrementAndGet();
                }

            });
        } else if (dtype == TFloat64.DTYPE) {
            Output<TFloat64> o = (Output<TFloat64>) input;
            AtomicInteger index = new AtomicInteger();
            if (debug) {
                o.data().scalars().forEach(f -> {
                    System.out.printf("%d). %f\n", index.incrementAndGet(), f.getDouble());
                });
            }
            index.set(0);
            o.data().scalars().forEach(f -> {
                assertEquals(expected[index.get()].doubleValue(), f.getDouble(), epsilon);
                if (!scalarExpected) {
                    index.incrementAndGet();
                }
            });
        } else if (dtype == TInt32.DTYPE) {
            Output<TInt32> o = (Output<TInt32>) input;
            AtomicInteger index = new AtomicInteger();
            if (debug) {
                o.data().scalars().forEach(f -> {
                    System.out.printf("%d). %d\n", index.incrementAndGet(), f.getInt());
                });
            }
            index.set(0);
            o.data().scalars().forEach(f -> {
                assertEquals(expected[index.get()].intValue(), f.getInt());
                if (!scalarExpected) {
                    index.incrementAndGet();
                }
            });
        } else if (dtype == TInt64.DTYPE) {
            Output<TInt64> o = (Output<TInt64>) input;
            AtomicInteger index = new AtomicInteger();
            if (debug) {
                o.data().scalars().forEach(f -> {
                    System.out.printf("%d). %d\n", index.incrementAndGet(), f.getLong());
                });
            }
            index.set(0);
            o.data().scalars().forEach(f -> {
                assertEquals(expected[index.get()].longValue(), f.getLong());
                if (!scalarExpected) {
                    index.incrementAndGet();
                }
            });
        }

    }

    @Override
    public <T extends TNumber> void evaluate(Output<T> expected, Output<T> input) {
        DataType dtype = input.asOutput().dataType();
        if (dtype == TFloat32.DTYPE) {
            Output<TFloat32> x = (Output<TFloat32>) expected;
            Output<TFloat32> o = (Output<TFloat32>) input;
            AtomicInteger index = new AtomicInteger();
            if (debug) {
                o.data().scalars().forEachIndexed((idx, f) -> {
                    System.out.printf("%d). %f <==> %f\n", index.incrementAndGet(), x.data().getFloat(idx), f.getFloat());
                });
            }
            index.set(0);
            o.data().scalars().forEachIndexed((idx, f) -> {
                assertEquals(x.data().getFloat(idx), f.getFloat(), epsilon);

            });
        } else if (dtype == TFloat64.DTYPE) {
            Output<TFloat64> x = (Output<TFloat64>) expected;
            Output<TFloat64> o = (Output<TFloat64>) input;
            AtomicInteger index = new AtomicInteger();
            if (debug) {
                o.data().scalars().forEachIndexed((idx, f) -> {
                    System.out.printf("%d). %f <==> %f\n", index.incrementAndGet(), x.data().getDouble(idx), f.getDouble());
                });
            }
            index.set(0);
            o.data().scalars().forEachIndexed((idx, f) -> {
                assertEquals(x.data().getDouble(idx), f.getDouble(), epsilon);
            });
        } else if (dtype == TInt32.DTYPE) {
            Output<TInt32> x = (Output<TInt32>) expected;
            Output<TInt32> o = (Output<TInt32>) input;
            AtomicInteger index = new AtomicInteger();
            if (debug) {
                o.data().scalars().forEachIndexed((idx, f) -> {
                    System.out.printf("%d). %d  <==>  %d\n", index.incrementAndGet(), x.data().getInt(idx), f.getInt());
                });
            }
            index.set(0);
            o.data().scalars().forEachIndexed((idx, f) -> {
                assertEquals(x.data().getInt(idx), f.getInt());
            });
        } else if (dtype == TInt64.DTYPE) {
            Output<TInt64> x = (Output<TInt64>) expected;
            Output<TInt64> o = (Output<TInt64>) input;
            AtomicInteger index = new AtomicInteger();
            if (debug) {
                o.data().scalars().forEachIndexed((idx, f) -> {
                    System.out.printf("%d). %d  <==> %d\n", index.incrementAndGet(), x.data().getLong(idx), f.getLong());
                });
            }
            index.set(0);
            o.data().scalars().forEachIndexed((idx, f) -> {
                assertEquals(x.data().getLong(idx), f.getLong());
            });
        }

    }

    public void print(PrintWriter writer, Output input) {
        DataType dtype = input.asOutput().dataType();
        if (dtype == TFloat32.DTYPE) {
            Output<TFloat32> o = (Output<TFloat32>) input;
            AtomicInteger index = new AtomicInteger();
            o.data().scalars().forEachIndexed((idx, f) -> {
                System.out.printf("%d). %f <==> %f\n", index.incrementAndGet(),  f.getFloat());
            });
        } else if (dtype == TFloat64.DTYPE) {
            Output<TFloat64> o = (Output<TFloat64>) input;
            AtomicInteger index = new AtomicInteger();
            o.data().scalars().forEachIndexed((idx, f) -> {
                System.out.printf("%d). %f <==> %f\n", index.incrementAndGet(), f.getDouble());
            });
        } else if (dtype == TInt32.DTYPE) {
            Output<TInt32> o = (Output<TInt32>) input;
            AtomicInteger index = new AtomicInteger();
            o.data().scalars().forEachIndexed((idx, f) -> {
                System.out.printf("%d). %d  <==>  %d\n", index.incrementAndGet(), f.getInt());
            });
        } else if (dtype == TInt64.DTYPE) {
            Output<TInt64> o = (Output<TInt64>) input;
            AtomicInteger index = new AtomicInteger();
            o.data().scalars().forEachIndexed((idx, f) -> {
                System.out.printf("%d). %d  <==> %d\n", index.incrementAndGet(), f.getLong());
            });
        } else {
            writer.println("Unexpected DataType: " + dtype);
        }
        writer.flush();
    }
}
