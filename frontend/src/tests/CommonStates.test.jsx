import React from 'react'
import { render, screen, fireEvent } from '@testing-library/react'
import { describe, it, expect, vi } from 'vitest'
import LoadingSpinner from '../components/common/LoadingSpinner'
import ErrorMessage from '../components/common/ErrorMessage'
import EmptyState from '../components/common/EmptyState'

describe('Common UI State Components', () => {
  it('renders LoadingSpinner with custom text and status role', () => {
    render(<LoadingSpinner text="Fetching digital twin models..." />)
    const status = screen.getByRole('status')
    expect(status).toHaveTextContent('Fetching digital twin models...')
  })

  it('renders ErrorMessage with retry button action', () => {
    const handleRetry = vi.fn()
    render(<ErrorMessage title="Connection Error" message="API unreachable" onRetry={handleRetry} />)

    expect(screen.getByRole('alert')).toBeInTheDocument()
    expect(screen.getByText('Connection Error')).toBeInTheDocument()
    expect(screen.getByText('API unreachable')).toBeInTheDocument()

    const retryBtn = screen.getByRole('button', { name: /retry/i })
    fireEvent.click(retryBtn)
    expect(handleRetry).toHaveBeenCalledOnce()
  })

  it('renders EmptyState with custom title and message', () => {
    render(<EmptyState title="No Records Available" message="Please adjust filters" />)
    expect(screen.getByText('No Records Available')).toBeInTheDocument()
    expect(screen.getByText('Please adjust filters')).toBeInTheDocument()
  })
})
