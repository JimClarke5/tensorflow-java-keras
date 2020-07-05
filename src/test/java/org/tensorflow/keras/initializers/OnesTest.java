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
import org.tensorflow.keras.utils.PrintUtils;
import org.tensorflow.op.Ops;
import org.tensorflow.tools.Shape;
import org.tensorflow.tools.buffer.DataBuffers;
import org.tensorflow.types.TBool;
import org.tensorflow.types.TFloat32;
import org.tensorflow.types.TFloat64;
import org.tensorflow.types.TInt32;
import org.tensorflow.types.TInt64;
import org.tensorflow.types.TString;
import org.tensorflow.types.TUint8;

/**
 *
 * @author Jim Clarke
 */
public class OnesTest {

    private static final double EPSILON = 1e-7;
    private static final float EPSILON_F = 1e-7f;

    private int counter;

    public OnesTest() {
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
     * Test of call method, of class Ones.
     */
    @Test
    public void testCallUInt() {
        byte[] actual = {0, 0, 0, 0};
        byte[] expected = {1, 1, 1, 1}; // init to ones to make sure they all changet to zero
        try (EagerSession session = EagerSession.create()) {
            Ops tf = Ops.create(session);
            Shape shape = Shape.of(2, 2);
            Ones<TUint8> instance = new Ones<>(tf);
            Operand<TUint8> operand = instance.call(tf.constant(shape.asArray()), TUint8.DTYPE);
            operand.asTensor().data().read(DataBuffers.of(actual));
            // PrintUtils.print(operand.asTensor());
            assertArrayEquals(expected, actual);
        }
    }

    /**
     * Test of call method, of class Ones.
     */
    @Test
    public void testCallInt() {
        int[] actual = {0, 0, 0, 0};
        int[] expected = {1, 1, 1, 1}; // init to ones to make sure they all changet to zero
        try (EagerSession session = EagerSession.create()) {
            Ops tf = Ops.create(session);
            Shape shape = Shape.of(2, 2);
            Ones<TInt32> instance = new Ones<>(tf);
            Operand<TInt32> operand = instance.call(tf.constant(shape.asArray()), TInt32.DTYPE);
            operand.asTensor().data().read(DataBuffers.of(actual));
            // PrintUtils.print(operand.asTensor());
            assertArrayEquals(expected, actual);
        }
    }

    /**
     * Test of call method, of class Ones.
     */
    @Test
    public void testCallLong() {
        long[] actual = {0, 0, 0, 0};
        long[] expected = {1, 1, 1, 1}; // init to ones to make sure they all changet to zero
        try (EagerSession session = EagerSession.create()) {
            Ops tf = Ops.create(session);
            Shape shape = Shape.of(2, 2);
            Ones<TInt64> instance = new Ones<>(tf);
            Operand<TInt64> operand = instance.call(tf.constant(shape.asArray()), TInt64.DTYPE);
            operand.asTensor().data().read(DataBuffers.of(actual));
            // PrintUtils.print(operand.asTensor());
            assertArrayEquals(expected, actual);
        }
    }

    /**
     * Test of call method, of class Ones.
     */
    @Test
    public void testCallFloat() {
        float[] actual = {0.F, 0.F, 0.F, 0.F};
        float[] expected = {1.f, 1.f, 1.f, 1.f};
        try (EagerSession session = EagerSession.create()) {
            Ops tf = Ops.create(session);
            Shape shape = Shape.of(2, 2);
            Ones<TFloat32> instance = new Ones<>(tf);
            Operand<TFloat32> operand = instance.call(tf.constant(shape.asArray()), TFloat32.DTYPE);
            operand.asTensor().data().read(DataBuffers.of(actual));
            PrintUtils.print(operand.asTensor());
            assertArrayEquals(expected, actual, EPSILON_F);
        }
    }

    /**
     * Test of call method, of class Ones.
     */
    @Test
    public void testCallDouble() {
        double[] actual = {0., 0., 0., 0.};
        double[] expected = {1., 1., 1., 1.};
        try (EagerSession session = EagerSession.create()) {
            Ops tf = Ops.create(session);
            Shape shape = Shape.of(2, 2);

            Ones<TFloat64> instance = new Ones<>(tf);
            Operand<TFloat64> operand = instance.call(tf.constant(shape.asArray()), TFloat64.DTYPE);
            operand.asTensor().data().read(DataBuffers.of(actual));
            PrintUtils.print(operand.asTensor());
            assertArrayEquals(expected, actual, EPSILON);
        }
    }

    /**
     * Test of call method, of class Ones.
     */
    @Test
    public void testCallString() {
        try (EagerSession session = EagerSession.create()) {
            Ops tf = Ops.create(session);
            Shape shape = Shape.of(2, 2);

            Ones<TString> instance = new Ones<>(tf);
            Operand<TString> operand = instance.call(tf.constant(shape.asArray()), TString.DTYPE);
            fail("AssertionError should have been thrown for TString");
        } catch (AssertionError expected) {
        }
    }

    @Test
    public void testCallBool() {
        boolean[] expected = {true, true, true, true};
        boolean[] actual = {false, false, false, false};
        try (EagerSession session = EagerSession.create()) {
            Ops tf = Ops.create(session);
            Shape shape = Shape.of(2, 2);

            Ones<TBool> instance = new Ones<>(tf);
            Operand<TBool> operand = instance.call(tf.constant(shape.asArray()), TBool.DTYPE);
            //counter = 0;
            //operand.asTensor().data().scalars().forEach(s -> {/*counter++;*/ assertFalse(s.getBoolean());});
            //assertEquals(counter, 2*2);
            operand.asTensor().data().read(DataBuffers.of(actual));
            assertArrayEquals(expected, actual);
        }
    }

}
