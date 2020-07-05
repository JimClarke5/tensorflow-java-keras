/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
=======================================================================*/
package org.tensorflow.keras.initializers;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
import org.tensorflow.EagerSession;
import org.tensorflow.Operand;
import org.tensorflow.op.Ops;
import org.tensorflow.tools.Shape;
import org.tensorflow.tools.buffer.DataBuffers;
import org.tensorflow.types.TFloat32;
import org.tensorflow.types.TFloat64;
import org.tensorflow.types.TInt32;
import org.tensorflow.types.TInt64;
import org.tensorflow.types.TUint8;

/**
 *
 * @author Jim Clarke
 */
public class InitializerFunctionTest {

    private static final double EPSILON = 1e-7;
    private static final float EPSILON_F = 1e-7f;

    public InitializerFunctionTest() {
    }

    @BeforeAll
    public static void setUpClass() {
    }

    @AfterAll
    public static void tearDownClass() {
    }

    @BeforeEach
    public void setUp() {
    }

    @AfterEach
    public void tearDown() {
    }

    /**
     * Test of call method, of class OptimizerFunction.
     */
    @Test
    public void testLambdaCallFloat() {
        float[] floats = {12345.0F};
        try (EagerSession session = EagerSession.create()) {
            Ops tf = Ops.create(session);
            Shape shape = Shape.of(1);

            // Test float
            InitializerFunction<TFloat32> func = (dims, dtype) -> {
                return tf.dtypes.cast(tf.constant(floats[0]), dtype);
            };
            Operand<TFloat32> operand = func.call(tf.constant(shape.asArray()), TFloat32.DTYPE);
            float[] actual = new float[floats.length];
            operand.asTensor().data().read(DataBuffers.of(actual));
            assertArrayEquals(floats, actual, EPSILON_F);

        }
    }

    /**
     * Test of call method, of class OptimizerFunction.
     */
    @Test
    public void testLambdaCallUInt8() {
        byte[] bytes = {0x15};
        try (EagerSession session = EagerSession.create()) {
            Ops tf = Ops.create(session);
            Shape shape = Shape.of(1);

            // Test float
            InitializerFunction<TUint8> func = (dims, dtype) -> {
                return tf.dtypes.cast(tf.constant(bytes[0]), dtype);
            };
            Operand<TUint8> operand = func.call(tf.constant(shape.asArray()), TUint8.DTYPE);
            byte[] actual = new byte[bytes.length];
            operand.asTensor().data().read(DataBuffers.of(actual));
            assertArrayEquals(bytes, actual);

        }
    }

    /**
     * Test of call method, of class OptimizerFunction.
     */
    @Test
    public void testLambdaCallInt() {
        int[] ints = {12345};
        try (EagerSession session = EagerSession.create()) {
            Ops tf = Ops.create(session);
            Shape shape = Shape.of(1);

            // Test float
            InitializerFunction<TInt32> func = (dims, dtype) -> {
                return tf.dtypes.cast(tf.constant(ints[0]), dtype);
            };
            Operand<TInt32> operand = func.call(tf.constant(shape.asArray()), TInt32.DTYPE);
            int[] actual = new int[ints.length];
            operand.asTensor().data().read(DataBuffers.of(actual));
            assertArrayEquals(ints, actual);

        }
    }

    /**
     * Test of call method, of class OptimizerFunction.
     */
    @Test
    public void testLambdaCallLong() {
        long[] longs = {12345L};
        try (EagerSession session = EagerSession.create()) {
            Ops tf = Ops.create(session);
            Shape shape = Shape.of(1);

            // Test float
            InitializerFunction<TInt64> func = (dims, dtype) -> {
                return tf.dtypes.cast(tf.constant(longs[0]), dtype);
            };
            Operand<TInt64> operand = func.call(tf.constant(shape.asArray()), TInt64.DTYPE);
            long[] actual = new long[longs.length];
            operand.asTensor().data().read(DataBuffers.of(actual));
            assertArrayEquals(longs, actual);

        }
    }

    public void testLambdaCallDouble() {
        double[] doubles = {Math.PI};
        try (EagerSession session = EagerSession.create()) {
            Ops tf = Ops.create(session);
            Shape shape = Shape.of(1);

            // Test float
            InitializerFunction<TFloat64> func = (dims, dtype) -> {
                return tf.dtypes.cast(tf.constant(doubles[0]), dtype);
            };
            Operand<TFloat64> operand = func.call(tf.constant(shape.asArray()), TFloat64.DTYPE);
            double[] actual = new double[doubles.length];
            operand.asTensor().data().read(DataBuffers.of(actual));
            assertArrayEquals(doubles, actual, EPSILON_F);

        }
    }

}
