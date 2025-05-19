package com.betool.ui;

import javax.swing.*;
import javax.swing.border.AbstractBorder;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.geom.RoundRectangle2D;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

public class BELibPanel extends JPanel {
    private JTextField beNameField;
    private JSlider windowSizeSlider;
    private JComboBox<String> editTypeCombo;
    private JTextField locStartField;
    private JTextField locEndField;

    public BELibPanel() {
        setLayout(new BoxLayout(this, BoxLayout.Y_AXIS));
        setOpaque(false);
        setBorder(BorderFactory.createEmptyBorder(10, 20, 20, 20));

        // 创建主面板
        JPanel mainPanel = new JPanel();
        mainPanel.setLayout(new BoxLayout(mainPanel, BoxLayout.Y_AXIS));
        mainPanel.setOpaque(false);
        mainPanel.setBorder(new CustomRoundedBorder(Color.BLACK, 1, 5));

        // 创建输入面板
        JPanel inputPanel = new JPanel(new GridBagLayout());
        inputPanel.setOpaque(false);
        GridBagConstraints gbc = new GridBagConstraints();
        gbc.gridx = 0;
        gbc.gridy = 0;
        gbc.insets = new Insets(5, 5, 5, 5);
        gbc.anchor = GridBagConstraints.LINE_START;

        // 第一行：BE Name
        gbc.gridy++;
        gbc.weightx = 0;
        inputPanel.add(createLabel("BE Name:"), gbc);

        gbc.gridx = 1;
        gbc.weightx = 1.0;
        gbc.fill = GridBagConstraints.HORIZONTAL;
        beNameField = new JTextField(20);
        beNameField.setBorder(new CustomRoundedBorder(Color.BLACK, 1, 5));
        inputPanel.add(beNameField, gbc);

        // 第二行：Window Size
        gbc.gridx = 0;
        gbc.gridy++;
        gbc.weightx = 0;
        inputPanel.add(createLabel("Window Size:"), gbc);

        gbc.gridx = 1;
        gbc.weightx = 1.0;
        gbc.fill = GridBagConstraints.HORIZONTAL;
        windowSizeSlider = new JSlider(JSlider.HORIZONTAL, 0, 20, 10);
        windowSizeSlider.setMajorTickSpacing(5);
        windowSizeSlider.setMinorTickSpacing(1);
        windowSizeSlider.setPaintTicks(true);
        windowSizeSlider.setPaintLabels(true);
        inputPanel.add(windowSizeSlider, gbc);

        // 第三行：Edit Type
        gbc.gridx = 0;
        gbc.gridy++;
        gbc.weightx = 0;
        inputPanel.add(createLabel("Edit Type:"), gbc);

        gbc.gridx = 1;
        gbc.weightx = 1.0;
        gbc.fill = GridBagConstraints.HORIZONTAL;
        String[] editTypes = {"C-to-T", "A-to-G", "Multiple", "Other"};
        editTypeCombo = new JComboBox<>(editTypes);
        editTypeCombo.setRenderer(new DefaultListCellRenderer() {
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
        inputPanel.add(editTypeCombo, gbc);

        // 第四行：LOC (Start and End)
        gbc.gridx = 0;
        gbc.gridy++;
        gbc.weightx = 0;
        inputPanel.add(createLabel("LOC:"), gbc);

        // 修改：将两个输入框的weightx设为相同值，使它们宽度一致
        gbc.gridx = 1;
        gbc.weightx = 0.5;
        gbc.fill = GridBagConstraints.HORIZONTAL;
        locStartField = new JTextField(10);
        locStartField.setBorder(new CustomRoundedBorder(Color.BLACK, 1, 5));
        locStartField.setToolTipText("起始位置");
        inputPanel.add(locStartField, gbc);

        gbc.gridx = 2;
        gbc.weightx = 0;
        inputPanel.add(createLabel("--"), gbc);

        gbc.gridx = 3;
        gbc.weightx = 0.5; // 与startField的weightx相同
        gbc.fill = GridBagConstraints.HORIZONTAL;
        locEndField = new JTextField(10);
        locEndField.setBorder(new CustomRoundedBorder(Color.BLACK, 1, 5));
        locEndField.setToolTipText("结束位置");
        inputPanel.add(locEndField, gbc);

        // 第五行：保存按钮
        JPanel buttonPanel = new JPanel(new FlowLayout(FlowLayout.CENTER));
        buttonPanel.setOpaque(false);

        JButton saveButton = new JButton("Save");
        saveButton.setFont(new Font("Arial", Font.BOLD, 14));
        saveButton.setForeground(Color.BLACK);
        saveButton.setBorder(new CustomRoundedBorder(Color.BLACK, 1, 5));
        saveButton.setBackground(new Color(180, 200, 255, 220));
        saveButton.setPreferredSize(new Dimension(120, 40));
        saveButton.addActionListener(this::saveSettings);

        buttonPanel.add(saveButton);

        // 添加所有面板
        mainPanel.add(inputPanel);
        mainPanel.add(Box.createVerticalStrut(15));
        mainPanel.add(buttonPanel);

        // 添加主面板到当前面板
        add(mainPanel);
        add(Box.createVerticalGlue());
    }

    private JLabel createLabel(String text) {
        JLabel label = new JLabel(text);
        label.setFont(new Font("Arial", Font.PLAIN, 14));
        label.setForeground(Color.BLACK);
        return label;
    }

    private void saveSettings(ActionEvent e) {
        String beName = beNameField.getText();
        int windowSize = windowSizeSlider.getValue();
        String editType = (String) editTypeCombo.getSelectedItem();

        int startPos = 0;
        int endPos = 0;

        try {
            startPos = Integer.parseInt(locStartField.getText());
            endPos = Integer.parseInt(locEndField.getText());

            if (startPos >= endPos) {
                JOptionPane.showMessageDialog(this, "起始位置必须小于结束位置", "输入错误", JOptionPane.ERROR_MESSAGE);
                return;
            }
        } catch (NumberFormatException ex) {
            JOptionPane.showMessageDialog(this, "请输入有效的数值", "输入错误", JOptionPane.ERROR_MESSAGE);
            return;
        }

        // 验证BE名称
        if (beName.isEmpty()) {
            JOptionPane.showMessageDialog(this, "请输入BE名称", "输入错误", JOptionPane.ERROR_MESSAGE);
            return;
        }

        // 保存设置
        try {
            JFileChooser fileChooser = new JFileChooser();
            fileChooser.setDialogTitle("保存设置");
            fileChooser.setSelectedFile(new File(beName + ".txt"));

            int userSelection = fileChooser.showSaveDialog(this);

            if (userSelection == JFileChooser.APPROVE_OPTION) {
                File fileToSave = fileChooser.getSelectedFile();
                try (FileWriter writer = new FileWriter(fileToSave)) {
                    writer.write("BE Name: " + beName + "\n");
                    writer.write("Window Size: " + windowSize + "\n");
                    writer.write("Edit Type: " + editType + "\n");
                    writer.write("LOC Start: " + startPos + "\n");
                    writer.write("LOC End: " + endPos + "\n");

                    JOptionPane.showMessageDialog(this, "设置已保存到 " + fileToSave.getAbsolutePath(),
                            "保存成功", JOptionPane.INFORMATION_MESSAGE);
                }
            }
        } catch (IOException ex) {
            JOptionPane.showMessageDialog(this, "保存失败: " + ex.getMessage(), "错误", JOptionPane.ERROR_MESSAGE);
            ex.printStackTrace();
        }
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
            JFrame frame = new JFrame("碱基编辑器库设置");
            frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
            frame.setSize(500, 350);
            frame.setLocationRelativeTo(null);

            BELibPanel panel = new BELibPanel();
            frame.setContentPane(panel);
            frame.setVisible(true);
        });
    }
}