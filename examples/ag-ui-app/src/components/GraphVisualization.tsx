import React from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  TimeScale,
  ChartOptions,
  ChartData,
  TooltipItem,
} from 'chart.js';
import { Line } from 'react-chartjs-2';
import 'chartjs-adapter-date-fns';
import './GraphVisualization.css';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  TimeScale
);

interface GraphData {
  title: string;
  query?: string;
  data: {
    result: Array<{
      metric: Record<string, string>;
      values: Array<[number, string]>;
    }>;
  };
  metadata?: {
    timestamp?: number;
    source?: string;
    start_time?: string;
    end_time?: string;
    step?: string;
  };
}

interface GraphVisualizationProps {
  data: GraphData;
}

const GraphVisualization: React.FC<GraphVisualizationProps> = ({ data }) => {
  const { title, query, data: graphData, metadata } = data;

  // Generate colors for different series (Grafana-like palette)
  const generateColor = (index: number) => {
    const colors = [
      '#7EB26D', // Green
      '#EAB839', // Yellow
      '#6ED0E0', // Light Blue
      '#EF843C', // Orange
      '#E24D42', // Red
      '#1F78C1', // Blue
      '#BA43A9', // Purple
      '#705DA0', // Dark Purple
      '#508642', // Dark Green
      '#CCA300', // Dark Yellow
    ];
    return colors[index % colors.length];
  };

  // Generate series label from metric
  const generateSeriesLabel = (metric: Record<string, string>) => {
    // Remove common labels like __name__ and job for cleaner display
    const filteredMetric = Object.entries(metric).filter(
      ([key]) => !['__name__', 'job'].includes(key)
    );
    
    if (filteredMetric.length === 0) {
      return metric.__name__ || 'Series';
    }
    
    return filteredMetric
      .map(([key, value]) => `${key}="${value}"`)
      .join(', ');
  };

  // Prepare Chart.js data
  const prepareChartData = (): ChartData<'line'> => {
    if (!graphData?.result || graphData.result.length === 0) {
      return { datasets: [] };
    }

    const datasets = graphData.result.map((series, index) => {
      const color = generateColor(index);
      const label = generateSeriesLabel(series.metric);
      
      const dataPoints = series.values?.map(([timestamp, value]) => ({
        x: timestamp * 1000, // Convert to milliseconds
        y: parseFloat(value),
      })) || [];

      return {
        label,
        data: dataPoints,
        borderColor: color,
        backgroundColor: color + '20', // Add transparency
        borderWidth: 2,
        fill: false,
        tension: 0.1,
        pointRadius: 2,
        pointHoverRadius: 4,
      };
    });

    return { datasets };
  };

  // Chart.js options (Grafana-like styling)
  const chartOptions: ChartOptions<'line'> = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: {
      mode: 'index',
      intersect: false,
    },
    plugins: {
      legend: {
        position: 'bottom',
        labels: {
          usePointStyle: true,
          padding: 15,
          font: {
            size: 12,
          },
        },
      },
      tooltip: {
        mode: 'index',
        intersect: false,
        backgroundColor: 'rgba(0, 0, 0, 0.8)',
        titleColor: '#fff',
        bodyColor: '#fff',
        borderColor: '#666',
        borderWidth: 1,
        callbacks: {
          title: (context: TooltipItem<'line'>[]) => {
            const date = new Date(context[0].parsed.x);
            return date.toLocaleString();
          },
          label: (context: TooltipItem<'line'>) => {
            return `${context.dataset.label}: ${context.parsed.y}`;
          },
        },
      },
    },
    scales: {
      x: {
        type: 'time',
        time: {
          displayFormats: {
            minute: 'HH:mm',
            hour: 'HH:mm',
            day: 'MMM dd',
          },
        },
        grid: {
          color: 'rgba(0, 0, 0, 0.1)',
        },
        ticks: {
          color: '#666',
        },
      },
      y: {
        grid: {
          color: 'rgba(0, 0, 0, 0.1)',
        },
        ticks: {
          color: '#666',
        },
      },
    },
  };

  const chartData = prepareChartData();

  if (!graphData?.result || graphData.result.length === 0) {
    return (
      <div className="graph-visualization">
        <div className="graph-container">
          <div className="graph-header">
            <h4>{title}</h4>
            {query && <code className="query">{query}</code>}
          </div>
          <div className="no-data">No data available</div>
        </div>
      </div>
    );
  }

  return (
    <div className="graph-visualization">
      <div className="graph-container">
        <div className="graph-header">
          <h4>{title}</h4>
          {query && <code className="query">{query}</code>}
        </div>
        
        <div className="chart-container">
          <Line data={chartData} options={chartOptions} />
        </div>
        
        {metadata && (
          <div className="graph-metadata">
            <small>
              Source: {metadata.source || 'Unknown'} | 
              Generated: {metadata.timestamp ? new Date(metadata.timestamp * 1000).toLocaleString() : 'Unknown'}
              {metadata.start_time && metadata.end_time && (
                <> | Range: {new Date(metadata.start_time).toLocaleString()} - {new Date(metadata.end_time).toLocaleString()}</>
              )}
            </small>
          </div>
        )}
      </div>
    </div>
  );
};

export default GraphVisualization;