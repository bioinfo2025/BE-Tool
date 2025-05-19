package com.betool.ui;

import javax.swing.*;
import javax.swing.border.AbstractBorder;
import javax.swing.table.DefaultTableModel;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.geom.RoundRectangle2D;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.net.URL;
import javax.imageio.ImageIO;

public class SgRNADesignPanel extends JPanel {
    private JTextField targetSeqField;
    private JTextField pamField;
    private JComboBox<String> editingTypeCombo;
    private JSlider windowSizeSlider;
    private JTable resultTable;
    private DefaultTableModel tableModel;

    public SgRNADesignPanel() {
        setLayout(new BoxLayout(this, BoxLayout.Y_AXIS));
        setOpaque(false);
        setBorder(BorderFactory.createEmptyBorder(10, 20, 20, 20));

        // 创建输入面板，使用GridBagLayout实现对齐
        JPanel inputPanel = new JPanel(new GridBagLayout());
        inputPanel.setOpaque(false);
        GridBagConstraints gbc = new GridBagConstraints();
        gbc.gridx = 0;
        gbc.gridy = 0;
        gbc.insets = new Insets(5, 5, 5, 5);
        gbc.anchor = GridBagConstraints.LINE_START;

        // 第一行：Target Sequence
        gbc.gridy++;
        gbc.weightx = 0;
        inputPanel.add(createLabel("Target Sequence:"), gbc);

        gbc.gridx = 1;
        gbc.weightx = 1.0;
        gbc.fill = GridBagConstraints.HORIZONTAL;
        targetSeqField = new JTextField(30);
        targetSeqField.setBorder(new CustomRoundedBorder(Color.BLACK, 1, 5));
        inputPanel.add(targetSeqField, gbc);

        gbc.gridx = 2;
        gbc.weightx = 0;
        gbc.fill = GridBagConstraints.NONE;
        JButton uploadButton = new JButton("upload fasta file");
        uploadButton.setBorder(new CustomRoundedBorder(new Color(50, 150, 250), 1, 5));
        uploadButton.setBackground(new Color(230, 240, 255, 220));
        uploadButton.setForeground(Color.BLACK);
        uploadButton.addActionListener(e -> {
            JFileChooser fileChooser = new JFileChooser();
            if (fileChooser.showOpenDialog(this) == JFileChooser.APPROVE_OPTION) {
                targetSeqField.setText(fileChooser.getSelectedFile().getPath());
            }
        });
        inputPanel.add(uploadButton, gbc);

        // 第二行：PAM Sequence
        gbc.gridx = 0;
        gbc.gridy++;
        gbc.weightx = 0;
        gbc.fill = GridBagConstraints.NONE;
        inputPanel.add(createLabel("PAM Sequence:"), gbc);

        gbc.gridx = 1;
        gbc.weightx = 1.0;
        gbc.fill = GridBagConstraints.HORIZONTAL;
        pamField = new JTextField("NGG", 10);
        pamField.setBorder(new CustomRoundedBorder(Color.BLACK, 1, 5));
        inputPanel.add(pamField, gbc);

        // 第三行：Editing Type
        gbc.gridx = 0;
        gbc.gridy++;
        gbc.weightx = 0;
        gbc.fill = GridBagConstraints.NONE;
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
        gbc.fill = GridBagConstraints.NONE;
        inputPanel.add(createLabel("Window Size:"), gbc);

        gbc.gridx = 1;
        gbc.weightx = 1.0;
        gbc.fill = GridBagConstraints.HORIZONTAL;
        windowSizeSlider = new JSlider(JSlider.HORIZONTAL, 0, 10, 5);
        windowSizeSlider.setMajorTickSpacing(1);
        windowSizeSlider.setPaintTicks(true);
        windowSizeSlider.setPaintLabels(true);
        inputPanel.add(windowSizeSlider, gbc);

        // 设计按钮 - 单独一行并居中
        JPanel buttonPanel = new JPanel(new FlowLayout(FlowLayout.CENTER));
        buttonPanel.setOpaque(false);

        JButton designButton = new JButton("Design sgRNA");
        designButton.setFont(new Font("Arial", Font.BOLD, 14));
        designButton.setForeground(Color.BLACK);
        designButton.setBorder(new CustomRoundedBorder(Color.BLACK, 1, 5));
        designButton.setBackground(new Color(180, 200, 255, 220));
        designButton.setPreferredSize(new Dimension(180, 40));
        designButton.addActionListener(this::designSgRNA);

        buttonPanel.add(designButton);

        // 结果区域 - 使用表格展示
        JPanel resultPanel = new JPanel(new BorderLayout());
        resultPanel.setOpaque(false);
        resultPanel.add(createLabel("Result:"), BorderLayout.NORTH);

        // 定义表格模型（2列：序号、sgRNA序列）
        String[] columnNames = {"#", "sgRNA Sequence","BE"};
        tableModel = new DefaultTableModel(columnNames, 0);
        resultTable = new JTable(tableModel);
        resultTable.setRowHeight(30);
        resultTable.setBorder(new CustomRoundedBorder(Color.GRAY, 1, 5));
        resultTable.setShowHorizontalLines(true);
        resultTable.setShowVerticalLines(false);
        resultTable.setFont(new Font("Arial", Font.PLAIN, 14));
        resultTable.getColumnModel().getColumn(0).setPreferredWidth(50);
        resultTable.getColumnModel().getColumn(1).setPreferredWidth(300);

        // 添加滚动条
        JScrollPane scrollPane = new JScrollPane(resultTable);
        scrollPane.setBorder(null);
        scrollPane.getViewport().setOpaque(false);
        scrollPane.setOpaque(false);
        scrollPane.setMaximumSize(new Dimension(Integer.MAX_VALUE, 200));
        resultPanel.add(scrollPane, BorderLayout.CENTER);

        // 添加所有面板到主面板
        add(inputPanel);
        add(Box.createVerticalStrut(15));
        add(buttonPanel);
        add(Box.createVerticalStrut(15));
        add(resultPanel);
        add(Box.createVerticalGlue());
    }

    private JLabel createLabel(String text) {
        JLabel label = new JLabel(text);
        label.setFont(new Font("Arial", Font.PLAIN, 14));
        label.setForeground(Color.BLACK);
        return label;
    }

    // 设计sgRNA按钮的监听方法
    private void designSgRNA(ActionEvent e) {
        // 清空表格数据
        tableModel.setRowCount(0);

        String targetSeq = targetSeqField.getText();
        String pamSeq = pamField.getText();
        String editingType = (String) editingTypeCombo.getSelectedItem();
        int windowSize = windowSizeSlider.getValue();

        // 验证输入
        if (targetSeq.isEmpty()) {
            JOptionPane.showMessageDialog(this, "请输入Target Sequence", "提示", JOptionPane.WARNING_MESSAGE);
            return;
        }

        // 模拟设计sgRNA（实际应用中这里会调用算法）
        try {
            // 显示加载提示
            JOptionPane.showMessageDialog(this, "正在设计sgRNA...\n" +
                    "Target: " + targetSeq + "\n" +
                    "PAM: " + pamSeq + "\n" +
                    "Type: " + editingType + "\n" +
                    "Window: " + windowSize, "处理中", JOptionPane.INFORMATION_MESSAGE);
            //AAAGTTTAATAACTGTCGCT
            String[] sgRNAs ={"AAAGTTTAATAACTGTCGCT","AAGTTTAAATAACTGTCGCT","AGTTTAATAACTGTCGCTGA","GTTTAATAACTGTCGCGTGA","TTTAATAACTGTCGCTGATG"};
            String[] BES = {"ABE","ABE","ABE","ABE","ABE-CP1041"};

            // 模拟生成3条sgRNA数据
            for (int i = 0; i < 5; i++) {
                String sgRNAId = "sgRNA-" + (i + 1);
                String sequence =sgRNAs[i];
                String position = (100 + i * 20) + "-" + (120 + i * 20);
                double score = 80 + Math.random() * 20;

                // 添加到表格
                tableModel.addRow(new Object[]{
                        i + 1,
                        sgRNAId + "\ngene sequence: " + sequence + "\nloac: " + position + "\nscore: " + String.format("%.1f", score),BES[i]
                });
            }

            // 显示结果提示
            JOptionPane.showMessageDialog(this, "success！BE-Tool find " + tableModel.getRowCount() + " candidate sequences。", "SUCCESS", JOptionPane.INFORMATION_MESSAGE);

        } catch (Exception ex) {
            // 错误处理
            JOptionPane.showMessageDialog(this, "FAIL: " + ex.getMessage(), "FAIL", JOptionPane.ERROR_MESSAGE);
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

    // 背景面板类
    public static class BackgroundPanel extends JPanel {
        private BufferedImage backgroundImage;
        private boolean scaleImage;

        public BackgroundPanel(String imagePath, boolean scaleImage) {
            this.scaleImage = scaleImage;

            try {
                File imageFile = new File(imagePath);
                if (imageFile.exists()) {
                    backgroundImage = ImageIO.read(imageFile);
                    System.out.println("成功加载背景图片");
                    return;
                }

                URL imageUrl = getClass().getResource("/" + imagePath);
                if (imageUrl != null) {
                    backgroundImage = ImageIO.read(imageUrl);
                    System.out.println("成功从类路径加载背景图片");
                    return;
                }

                System.err.println("背景图片不存在: " + imagePath);

            } catch (IOException e) {
                System.err.println("加载背景图片失败: " + e.getMessage());
            }
        }

        @Override
        protected void paintComponent(Graphics g) {
            super.paintComponent(g);

            if (backgroundImage != null) {
                if (scaleImage) {
                    g.drawImage(backgroundImage, 0, 0, getWidth(), getHeight(), this);
                } else {
                    int x = (getWidth() - backgroundImage.getWidth()) / 2;
                    int y = (getHeight() - backgroundImage.getHeight()) / 2;
                    g.drawImage(backgroundImage, x, y, this);
                }
            } else {
                g.setColor(new Color(220, 230, 240));
                g.fillRect(0, 0, getWidth(), getHeight());
            }
        }
    }

    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> {
            JFrame frame = new JFrame("sgRNA Design Tool");
            frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
            frame.setSize(700, 500);
            frame.setLocationRelativeTo(null);

            BackgroundPanel backgroundPanel = new BackgroundPanel("background.png", true);
            backgroundPanel.setLayout(new BorderLayout());
            backgroundPanel.add(new SgRNADesignPanel(), BorderLayout.CENTER);

            frame.setContentPane(backgroundPanel);
            frame.setVisible(true);
        });
    }
}