package com.betool.ui;

import javax.swing.*;
import javax.swing.border.AbstractBorder;
import javax.swing.table.DefaultTableModel;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.geom.RoundRectangle2D;
import java.util.Random;

public class BEditorPanel extends JPanel {
    // 评估面板组件
    private JTextField sgRnaField;
    private JTextField targetSeqField;
    private JComboBox<String> editingTypeCombo;
    private JSlider windowSizeSlider;
    private JTable resultTable;
    private DefaultTableModel tableModel;

    public BEditorPanel() {
        setLayout(new BoxLayout(this, BoxLayout.Y_AXIS));
        setOpaque(false);
        setBorder(BorderFactory.createEmptyBorder(10, 20, 20, 20));

        // 创建评估面板
        JPanel evalPanel = new JPanel();
        evalPanel.setLayout(new BoxLayout(evalPanel, BoxLayout.Y_AXIS));
        evalPanel.setOpaque(false);


        // 创建评估输入面板
        JPanel inputPanel = new JPanel(new GridBagLayout());
        inputPanel.setOpaque(false);
        GridBagConstraints gbc = new GridBagConstraints();
        gbc.gridx = 0;
        gbc.gridy = 0;
        gbc.insets = new Insets(5, 5, 5, 5);
        gbc.anchor = GridBagConstraints.LINE_START;

        // 第一行：sgRNA
        gbc.gridy++;
        gbc.weightx = 0;
        inputPanel.add(createLabel("sgRNA:"), gbc);

        gbc.gridx = 1;
        gbc.weightx = 1.0;
        gbc.fill = GridBagConstraints.HORIZONTAL;
        sgRnaField = new JTextField(30);
        sgRnaField.setBorder(new CustomRoundedBorder(Color.BLACK, 1, 5));
        inputPanel.add(sgRnaField, gbc);

        // 第二行：Target Sequence
        gbc.gridx = 0;
        gbc.gridy++;
        gbc.weightx = 0;
        inputPanel.add(createLabel("Target Sequence:"), gbc);

        gbc.gridx = 1;
        gbc.weightx = 1.0;
        gbc.fill = GridBagConstraints.HORIZONTAL;
        targetSeqField = new JTextField(30);
        targetSeqField.setBorder(new CustomRoundedBorder(Color.BLACK, 1, 5));
        inputPanel.add(targetSeqField, gbc);

        // 第三行：Editing Type
        gbc.gridx = 0;
        gbc.gridy++;
        gbc.weightx = 0;
        inputPanel.add(createLabel("Editing Type:"), gbc);

        gbc.gridx = 1;
        gbc.weightx = 1.0;
        gbc.fill = GridBagConstraints.HORIZONTAL;
        String[] editingTypes = {"C-to-T", "A-to-G", "Multiple", "Other"};
        editingTypeCombo = new JComboBox<>(editingTypes);
        editingTypeCombo.setRenderer(new DefaultListCellRenderer() {
            @Override
            public Component getListCellRendererComponent(JList<?> list, Object value, int index,
                                                          boolean isSelected, boolean cellHasFocus) {
                Component c = super.getListCellRendererComponent(list, value, index, isSelected, cellHasFocus);
                if (c instanceof JLabel) {
                    JLabel label = (JLabel) c;
                    label.setBorder(BorderFactory.createEmptyBorder(5, 10, 5, 10));
                    label.setForeground(Color.BLACK);
                    if (isSelected) {
                        label.setBackground(new Color(50, 100, 200, 200));
                    } else {
                        label.setBackground(new Color(255, 255, 255, 220));
                    }
                }
                return c;
            }
        });
        inputPanel.add(editingTypeCombo, gbc);

        // 第四行：Window Size
        gbc.gridx = 0;
        gbc.gridy++;
        gbc.weightx = 0;
        inputPanel.add(createLabel("Window Size:"), gbc);

        gbc.gridx = 1;
        gbc.weightx = 1.0;
        gbc.fill = GridBagConstraints.HORIZONTAL;
        windowSizeSlider = new JSlider(JSlider.HORIZONTAL, 0, 10, 5);
        windowSizeSlider.setMajorTickSpacing(1);
        windowSizeSlider.setPaintTicks(true);
        windowSizeSlider.setPaintLabels(true);
        inputPanel.add(windowSizeSlider, gbc);

        // 输出按钮
        JPanel buttonPanel = new JPanel(new FlowLayout(FlowLayout.CENTER));
        buttonPanel.setOpaque(false);

        JButton evaluateButton = new JButton("SUBMIT");
        evaluateButton.setFont(new Font("Arial", Font.BOLD, 14));
        evaluateButton.setForeground(Color.BLACK);
        evaluateButton.setBorder(new CustomRoundedBorder(Color.BLACK, 1, 5));
        evaluateButton.setBackground(new Color(180, 200, 255, 220));
        evaluateButton.setPreferredSize(new Dimension(120, 40));
        evaluateButton.addActionListener(this::evaluateSgRNA);

        buttonPanel.add(evaluateButton);

        // 结果区域
        JPanel resultPanel = new JPanel(new BorderLayout());
        resultPanel.setOpaque(false);
        resultPanel.add(createLabel("Result:"), BorderLayout.NORTH);

        // 定义表格模型（4列：序号、编辑位置、预测得分、推荐BE）
        String[] columnNames = {"#", "Loc","score", "Recommended BE"};
        tableModel = new DefaultTableModel(columnNames, 0);
        resultTable = new JTable(tableModel);
        resultTable.setRowHeight(30);
        resultTable.setBorder(new CustomRoundedBorder(Color.GRAY, 1, 5));
        resultTable.setShowHorizontalLines(true);
        resultTable.setShowVerticalLines(false);
        resultTable.setFont(new Font("Arial", Font.PLAIN, 14));
        resultTable.getColumnModel().getColumn(0).setPreferredWidth(50);
        resultTable.getColumnModel().getColumn(1).setPreferredWidth(100);
        resultTable.getColumnModel().getColumn(2).setPreferredWidth(150);

        // 添加滚动条
        JScrollPane scrollPane = new JScrollPane(resultTable);
        scrollPane.setBorder(null);
        scrollPane.getViewport().setOpaque(false);
        scrollPane.setOpaque(false);
        scrollPane.setMaximumSize(new Dimension(Integer.MAX_VALUE, 200));
        resultPanel.add(scrollPane, BorderLayout.CENTER);

        // 添加所有面板到评估面板
        evalPanel.add(inputPanel);
        evalPanel.add(Box.createVerticalStrut(15));
        evalPanel.add(buttonPanel);
        evalPanel.add(Box.createVerticalStrut(15));
        evalPanel.add(resultPanel);

        // 添加评估面板到主面板
        add(evalPanel);
        add(Box.createVerticalGlue());
    }

    private JLabel createLabel(String text) {
        JLabel label = new JLabel(text);
        label.setFont(new Font("Arial", Font.PLAIN, 14));
        label.setForeground(Color.BLACK);
        return label;
    }

    // 评估sgRNA按钮的监听方法
    private void evaluateSgRNA(ActionEvent e) {
        // 清空表格数据
        tableModel.setRowCount(0);

        String sgRNA = sgRnaField.getText();
        String targetSeq = targetSeqField.getText();
        String editingType = (String) editingTypeCombo.getSelectedItem();
        int windowSize = windowSizeSlider.getValue();

        // 验证输入
        if (sgRNA.isEmpty() || targetSeq.isEmpty()) {
            JOptionPane.showMessageDialog(this, "Please enter sgRNA and Target Sequence", "WARNING", JOptionPane.WARNING_MESSAGE);
            return;
        }

        // 模拟评估sgRNA
        try {
            // 显示加载提示
            JOptionPane.showMessageDialog(this, "正在评估sgRNA...\n" +
                    "sgRNA: " + sgRNA + "\n" +
                    "Target: " + targetSeq + "\n" +
                    "Type: " + editingType + "\n" +
                    "Window: " + windowSize, "处理中", JOptionPane.INFORMATION_MESSAGE);

            // 模拟生成评估结果
            //String[] baseEditors = {"ABE", "ABE", "YE1-BE3", "HF1-BE3", "xCas9-BE3", "ABE8e"};
            String[] baseEditors = {"ABE", "ABE", "ABE", "ABE", "ABE", "ABE","ABE","ABE8e"};

            Random random = new Random();

            for (int i = 0; i < 7; i++) {
                double score = 60 + random.nextDouble() * 40;
                String recommendedBE = baseEditors[random.nextInt(baseEditors.length)];
                int j = i+1;

                // 添加到表格
                tableModel.addRow(new Object[]{
                        i+1,
                        "A"+j+"",
                        String.format("%.2f", score),
                        recommendedBE
                });
            }

            // 显示结果提示
            JOptionPane.showMessageDialog(this, "sgRNA评估完成！", "成功", JOptionPane.INFORMATION_MESSAGE);

        } catch (Exception ex) {
            // 错误处理
            JOptionPane.showMessageDialog(this, "sgRNA评估失败: " + ex.getMessage(), "错误", JOptionPane.ERROR_MESSAGE);
            ex.printStackTrace();
        }
    }

    private double calculateGCContent(String sequence) {
        // 计算GC含量
        int gcCount = 0;
        sequence = sequence.toUpperCase();
        for (char c : sequence.toCharArray()) {
            if (c == 'G' || c == 'C') {
                gcCount++;
            }
        }
        return (double) gcCount / sequence.length() * 100;
    }

    // 自定义圆角边框类
    private static class CustomRoundedBorder extends AbstractBorder {
        private final Color color;
        private final int thickness;
        private final int radius;
        private final Insets insets;
        private final BasicStroke stroke;
        private final int strokePad;
        private final RenderingHints hints;

        public CustomRoundedBorder(Color color, int thickness, int radius) {
            this.color = color;
            this.thickness = thickness;
            this.radius = radius;

            stroke = new BasicStroke(thickness);
            strokePad = thickness / 2;

            hints = new RenderingHints(
                    RenderingHints.KEY_ANTIALIASING,
                    RenderingHints.VALUE_ANTIALIAS_ON
            );

            int pad = radius + strokePad;
            int bottomPad = pad + 1;
            insets = new Insets(pad, pad, bottomPad, pad);
        }

        @Override
        public Insets getBorderInsets(Component c) {
            return insets;
        }

        @Override
        public void paintBorder(Component c, Graphics g, int x, int y, int width, int height) {
            Graphics2D g2 = (Graphics2D) g;

            Color originalColor = g2.getColor();
            Stroke originalStroke = g2.getStroke();
            RenderingHints originalHints = g2.getRenderingHints();

            g2.setRenderingHints(hints);
            g2.setColor(color);
            g2.setStroke(stroke);

            int actualWidth = width - (thickness * 2);
            int actualHeight = height - (thickness * 2);

            RoundRectangle2D.Double rect = new RoundRectangle2D.Double(
                    strokePad,
                    strokePad,
                    actualWidth,
                    actualHeight,
                    radius,
                    radius
            );
            g2.draw(rect);

            g2.setColor(originalColor);
            g2.setStroke(originalStroke);
            g2.setRenderingHints(originalHints);
        }
    }

    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> {
            JFrame frame = new JFrame("sgRNA评估工具");
            frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
            frame.setSize(600, 500);
            frame.setLocationRelativeTo(null);

            SgRNADesignPanel mainPanel = new SgRNADesignPanel();
            frame.setContentPane(mainPanel);
            frame.setVisible(true);
        });
    }
}